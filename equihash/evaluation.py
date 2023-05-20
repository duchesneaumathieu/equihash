import torch
from typing import List
from dataclasses import dataclass, field

@dataclass
class QuickResults:
    which: str
    steps: List[int] = field(default_factory=list)
    avg_recall_at_0: List[float] = field(default_factory=list)
    avg_recall_at_1: List[float] = field(default_factory=list)
    avg_recall_at_2: List[float] = field(default_factory=list)
    nb_unique_buckets: List[int] = field(default_factory=list)
    nb_documents: List[int] = field(default_factory=list)
    
    def describe(self, k):
        return (f'[{self.which} step:{self.steps[-1]}] '
                f'r0:{100*self.avg_recall_at_0[k]:.2f}%, '
                f'r1:{100*self.avg_recall_at_1[k]:.2f}%, '
                f'r2:{100*self.avg_recall_at_2[k]:.2f}%, '
                f'unique_bucket={self.nb_unique_buckets[k]:,}/{self.nb_documents[k]:,}')
    
    def evaluate_from_logits(self, step, logits):
        self.steps.append(step)
        codes = 0<logits
        hd = (codes[:,0]^codes[:,1]).sum(dim=1)
        self.avg_recall_at_0.append(float((hd==0).float().mean()))
        self.avg_recall_at_1.append(float((hd<=1).float().mean()))
        self.avg_recall_at_2.append(float((hd<=2).float().mean()))
        self.nb_unique_buckets.append(number_of_buckets(codes[:,0]))
        self.nb_documents.append(len(codes))
        return self
    
    def evaluate(self, step, net, loader, batch_size, nb_documents, seed=None):
        generator = torch.Generator(device=loader.device)
        if seed is not None: generator.manual_seed(seed)
        with torch.no_grad():
            logits = list()
            while nb_documents > 0:
                size = min(batch_size, nb_documents)
                nb_documents = nb_documents - size
                batch = loader.positive_batch(size=size, generator=generator)
                logits.append(net(batch))
            logits = torch.cat(logits, dim=0)
        return self.evaluate_from_logits(step, logits)
        
def number_of_buckets(codes):
    d = dict()
    for i, code in enumerate(codes.tolist()):
        k = tuple(code)
        if k in d: d[k].append(i)
        else: d[k] = [i]
    return len(d)
    
def saturation_ratio(logits, threshold: float):
    return float((threshold<logits.abs()).float().mean())
    