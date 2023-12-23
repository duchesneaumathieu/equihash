import torch
import numpy as np
from typing import List, Dict
from dataclasses import dataclass, field

def number_of_buckets(codes):
    d = dict()
    for i, code in enumerate(codes.tolist()):
        k = tuple(code)
        if k in d: d[k].append(i)
        else: d[k] = [i]
    return len(d)
    
def saturation_ratio(logits, threshold: float):
    return float((threshold<logits.abs()).float().mean())

def count_integers(array, n):
    #array of integer
    u, c = np.unique(array, return_counts=True)
    x = np.zeros(n+1, dtype=np.int64)
    x[u] = c
    return x

@dataclass
class QuickResult:
    nb_pairs: int
    positive_hamming_distances: List[int]
    negative_hamming_distances: List[int]
    positive_hamming_distances_median: int
    negative_hamming_distances_median: int
    nb_codes: int
    nb_unique_codes: int
    worst_entropies: Dict[int, float] = field(default_factory=dict)
    
    def __repr__(self):
        tp, fp = 0, 0
        rs, ps = list(), list()
        for ph, nh in zip(self.positive_hamming_distances[:3], self.negative_hamming_distances[:3]):
            tp += ph
            fp += nh
            rs.append(f'{tp / self.nb_pairs:.2f}')
            ps.append(f'{tp / (tp + fp):.2f}' if tp+fp > 0 else 'None')
        rp = ' - '.join([f'rp{i}:{r}:{p}' for i, (r, p) in enumerate(zip(rs, ps))])
        we = ' - '.join([f'we{i}:{e:.2f}' for i, e in self.worst_entropies.items()])
        m = f'pm:{self.positive_hamming_distances_median} - nm:{self.negative_hamming_distances_median}'
        u = f'u:{self.nb_unique_codes:,}/{self.nb_codes:,}'
        return ' - '.join([rp, we, m, u])
    
class QuickResults:
    def __init__(self, net, loader, batch_size, nb_documents, seed,
                 entropy_tuple_sizes=[1,2,3,4], max_nb_entropy_tuples=10_000):
        self.which = loader.which
        self.net = net
        self.loader = loader
        self.batch_size = batch_size
        self.nb_documents = nb_documents
        self.seed = seed
        self.entropy_tuple_sizes = entropy_tuple_sizes
        self.max_nb_entropy_tuples = max_nb_entropy_tuples
        self.hinge_triplet_labels = self.sample_hinge_triplet_labels()
        self.results = dict()
        
    def __getitem__(self, step):
        return self.results[step]
    
    def sample_hinge_triplet_labels(self):
        running_loader_state = self.loader.get_state()
        self.loader.manual_seed(self.seed)
        hinge_triplet_labels =  self.loader.labels_generator.generate_hinge_triplet(self.nb_documents, replacement=False)
        self.loader.set_state(running_loader_state)
        return hinge_triplet_labels
    
    def compute_logits(self):
        N, B = self.nb_documents, self.batch_size
        nb_batch = N//B + (N%B!=0)
        with torch.no_grad():
            logits = list()
            for i in range(nb_batch):
                a, b = i*B, min(N, (i+1)*B)
                labels = self.hinge_triplet_labels[a:b]
                batch = self.loader.batch_from_labels(labels)
                logits.append(self.net(batch))
            logits = torch.cat(logits, dim=0)
        return logits
    
    def evaluate_from_logits(self, logits):
        bs, *shape, nbits = logits.shape
        codes = 0<logits
        pos_hd = (codes[:,0]^codes[:,1]).sum(dim=1).cpu().numpy()
        neg_hd = (codes[:,0]^codes[:,2]).sum(dim=1).cpu().numpy()
        
        pos_hd_bins = count_integers(pos_hd, n=nbits)
        neg_hd_bins = count_integers(neg_hd, n=nbits)
        
        pos_hd_median = int(np.median(pos_hd))
        neg_hd_median = int(np.median(neg_hd))
        
        no_replacement_codes = torch.cat([codes[:,0], codes[:,2]], axis=0)
        nb_unique_codes = number_of_buckets(no_replacement_codes)
        
        #insert worst entropies code here
        
        return QuickResult(
            nb_pairs=len(logits),
            positive_hamming_distances=pos_hd_bins,
            negative_hamming_distances=neg_hd_bins,
            positive_hamming_distances_median=pos_hd_median,
            negative_hamming_distances_median=neg_hd_median,
            nb_codes=len(no_replacement_codes),
            nb_unique_codes=nb_unique_codes,
            worst_entropies={1:1, 2:2, 3:3, 4:4},
        )
    
    def evaluate(self, step):
        running_loader_state = self.loader.get_state()
        self.loader.manual_seed(self.seed)
        logits = self.compute_logits()
        self.loader.set_state(running_loader_state)
        
        result = self.evaluate_from_logits(logits)
        self.results[step] = result
        return result
    
    def state_dict(self):
        return {step:result.__dict__ for step, result in self.results.items()}
    
    def load_state_dict(self, state):
        for step, result in state.items():
            self.results[step] = QuickResult(**result)
        return self

    def describe(self, step):
        return f'[{self.which} - step:{step}] {self.results[step]}'
