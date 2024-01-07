import torch
import numpy as np

from typing import List, Tuple
from dataclasses import dataclass, field

from equihash.utils import timestamp
from equihash.utils.fingerprints import Fingerprints
from .evaluation import count_integers

def count_internal_collisions(ii, labels_ii, verbose=False):
    nb_cols = 0
    for n, (fingerprint, label) in enumerate(zip(ii.codes, labels_ii.codes)):
        if verbose and n%500_000==0: print(timestamp(f'{n:,} / {len(ii.codes):,}'), flush=True)
        ground_truth = set(labels_ii[label])
        #check that i > n to avoid double counts
        nb_cols += len([i for i in ii[fingerprint] if i > n and i not in ground_truth])
    if verbose: print(timestamp(f'{len(ii.codes):,} / {len(ii.codes):,}'), flush=True)
    return nb_cols

def get_queries_results(ii, queries, ground_truths, verbose=False):
    queries_results = list()
    for n, (ground_truth, query) in enumerate(zip(ground_truths, queries)):
        if verbose and n%500_000==0: print(timestamp(f'{n:,} / {len(queries):,}'), flush=True)
        queries_results.append(
            QueryResults(ground_truth=ground_truth, retrieved=ii[query])
        )
    if verbose: print(timestamp(f'{len(queries):,} / {len(queries):,}'), flush=True)
    return queries_results

def get_equihash_results(ii, labels_ii, positive_queries, negative_queries, ground_truths, verbose=False):
    if verbose: print(timestamp(f'Computing internal collision of a database with {len(ii.codes):,} items.'), flush=True)
    nb_database_internal_collisions = count_internal_collisions(ii, labels_ii, verbose=verbose)
    
    if verbose: print(timestamp(f'Computing positive queries results: {len(positive_queries):,} items...'), end='', flush=True)
    positive_queries_results = get_queries_results(ii, positive_queries, ground_truths, verbose=False)
    if verbose: print(f' done.', flush=True)
    
    if verbose: print(timestamp(f'Computing positive queries results: {len(negative_queries):,} items...'), end='', flush=True)
    negative_queries_results = get_queries_results(ii, negative_queries, ([] for q in negative_queries), verbose=False)
    if verbose: print(f' done.', flush=True)
    
    return EquihashResults(
        database_size = len(ii.codes),
        nb_database_internal_collisions = nb_database_internal_collisions,
        positive_queries_results = positive_queries_results,
        negative_queries_results = negative_queries_results,
    )

def get_binary_equihash_results(hinge_triplets_fingerprints, device='cpu'):
    nb_triplets, _, nbytes = hinge_triplets_fingerprints.shape
    triplets = torch.tensor(hinge_triplets_fingerprints, device=device)
    fingerprints = Fingerprints(device=device)
    pos_hd = fingerprints.uint8_hamming_distance(triplets[:,0], triplets[:,1])
    neg_hd = fingerprints.uint8_hamming_distance(triplets[:,0], triplets[:,2])
    
    pos_hd_bins = count_integers(pos_hd.cpu().numpy(), n=8*nbytes)
    neg_hd_bins = count_integers(neg_hd.cpu().numpy(), n=8*nbytes)
    
    return BinaryEquihashResults(
        positive_pairs_hamming_distances_histogram = pos_hd_bins.tolist(),
        negative_pairs_hamming_distances_histogram = neg_hd_bins.tolist(),
    )

@dataclass
class QueryResults:
    #the results for a single query
    ground_truth: List[int]
    retrieved: List[int]
    
    @property
    def is_hits(self):
        return bool(self.retrieved)
    
    @property
    def is_positive_query(self):
        return bool(self.ground_truth)
    
    @property
    def is_negative_query(self):
        return not self.is_positive_query
    
    @property
    def is_equiset_true_positive(self):
        return self.is_hits and self.is_positive_query
    
    @property
    def is_equiset_true_negative(self):
        return (not self.is_hits) and self.is_negative_query
    
    @property
    def is_equiset_false_positive(self):
        return self.is_hits and self.is_negative_query
    
    @property
    def is_equiset_false_negative(self):
        return (not self.is_hits) and self.is_positive_query
    
    @property
    def relevant_collisions(self):
        s = set(self.ground_truth)
        return [r for r in self.retrieved if r in s]
    
    @property
    def irrelevant_collisions(self):
        s = set(self.ground_truth)
        return [r for r in self.retrieved if r not in s]
    
    @property
    def nb_relevant(self):
        return len(self.ground_truth)
    
    @property
    def nb_collisions(self):
        return len(self.retrieved)
    
    @property
    def nb_relevant_collisions(self):
        return len(self.relevant_collisions)
    
    @property
    def nb_irrelevant_collisions(self):
        return len(self.irrelevant_collisions)
    
    @property
    def jaccard_index(self):
        #if both self.retrieved and self.ground_truth are 0, we set the Jaccard index to 1.
        intersection = self.nb_relevant_collisions
        union = self.nb_relevant + self.nb_irrelevant_collisions
        are_empty_sets = intersection==0 and union==0
        return 1.0 if are_empty_sets else intersection / union
    
    @property
    def is_perfect_retrieval(self):
        return set(self.ground_truth) == set(self.retrieved)

@dataclass
class EquiDictResults:
    positive_queries_results: List[QueryResults]
    
    def recall(self):
        nb_relevant = sum(r.nb_relevant for r in self.positive_queries_results)
        nb_relevant_collisions = sum(r.nb_relevant_collisions for r in self.positive_queries_results)
        return nb_relevant_collisions / nb_relevant
    
    def average_jaccard_index(self):
        jaccard_index_sum = sum(r.jaccard_index for r in self.positive_queries_results)
        return jaccard_index_sum / len(self.positive_queries_results)
    
    def perfect_retrieval_rate(self):
        nb_perfect_retrieval = sum(r.is_perfect_retrieval for r in self.positive_queries_results)
        return nb_perfect_retrieval / len(self.positive_queries_results)
    
    def print_results(self, prefix='', end='\n'):
        #TODO add confidence intervals
        print(f'{prefix}recall: {self.recall():.4f}', end=end)
        print(f'{prefix}average_jaccard_index: {self.average_jaccard_index():.4f}', end=end)
        print(f'{prefix}perfect_retrieval_rate: {self.perfect_retrieval_rate():.4f}', end=end)

@dataclass
class EquiSetResults:
    positive_queries_results: List[QueryResults]
    negative_queries_results: List[QueryResults]
    
    def true_positive_rate(self):
        nb_true_positive = sum(r.is_equiset_true_positive for r in self.positive_queries_results)
        return nb_true_positive / len(self.positive_queries_results)
    
    def false_positive_rate(self):
        nb_false_positive = sum(r.is_equiset_false_positive for r in self.negative_queries_results)
        return nb_false_positive / len(self.negative_queries_results)
    
    def print_results(self, prefix='', end='\n'):
        #TODO add confidence intervals
        print(f'{prefix}true_positive_rate: {self.true_positive_rate():.4f}', end=end)
        print(f'{prefix}false_positive_rate: {self.false_positive_rate():.4f}', end=end)

@dataclass
class EquihashResults:
    database_size: int
    nb_database_internal_collisions: int
    positive_queries_results: List[QueryResults]
    negative_queries_results: List[QueryResults]
    
    @property
    def equidict_results(self):
        return EquiDictResults(self.positive_queries_results)
    
    @property
    def equiset_results(self):
        return EquiSetResults(self.positive_queries_results, self.negative_queries_results)
    
    def print_equidict_results(self, prefix='', end='\n'):
        self.equidict_results.print_results(prefix=prefix, end=end)
    
    def print_equiset_results(self, prefix='', end='\n'):
        self.equiset_results.print_results(prefix=prefix, end=end)
    
    def print_results(self):
        max_nb_database_internal_collisions = self.database_size*(self.database_size-1)//2
        print(f'Database:')
        print(f'\tSize: {self.database_size:,}')
        print(f'\tInternal collisions: {self.nb_database_internal_collisions:,} / {max_nb_database_internal_collisions:,}')
        print()
        
        print('Equidict results:')
        self.print_equidict_results(prefix='\t')
        print()
        
        print('Equiset results:')
        self.print_equiset_results(prefix='\t')

@dataclass
class BinaryEquihashResults:
    positive_pairs_hamming_distances_histogram: List[int]
    negative_pairs_hamming_distances_histogram: List[int]
    at: Tuple[int] = (0, 1, 2, 3)
    
    def nb_positive_pairs(self):
        return sum(self.positive_pairs_hamming_distances_histogram)
    
    def nb_negative_pairs(self):
        return sum(self.negative_pairs_hamming_distances_histogram)
    
    def recalls(self):
        nb_positive_pairs = self.nb_positive_pairs()
        recalls = [self.positive_pairs_hamming_distances_histogram[i] / nb_positive_pairs for i in self.at]
        return np.cumsum(recalls).tolist()
    
    def fallouts(self):
        nb_negative_pairs = self.nb_negative_pairs()
        fallouts = [self.negative_pairs_hamming_distances_histogram[i] / nb_negative_pairs for i in self.at]
        return np.cumsum(fallouts).tolist()
    
    def print_results(self, prefix='', end='\n'):
        recalls = ' - '.join([f'r@{i}:{r:.4f}' for i, r in zip(self.at, self.recalls())])
        fallouts = ' - '.join([f'f@{i}:{f:.4f}' for i, f in zip(self.at, self.fallouts())])
        print(f'{prefix}{recalls}', end=end)
        print(f'{prefix}{fallouts}', end=end)
