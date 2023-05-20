import torch
import numpy as np
from scipy.special import comb
from itertools import combinations
from equihash.utils.more_torch import torch_lse, torch_logsumexp
log_sigmoid = torch.nn.LogSigmoid()


def unique_slice(tuples, start=None, end=None, step=None):
    unique = set()
    tuples = [t[start:end:step] for t in tuples]
    return [t for t in tuples if t not in unique and unique.add(t) is None]

def inverted_getitem(tuples):
    return {tuple(t):n for n, t in enumerate(tuples)}

def sub_index(tuples, sub_tuples, start=None, end=None, step=None):
    inv = inverted_getitem(sub_tuples)
    return [inv[t[start:end:step]] for t in tuples]

class PartialOuterProduct:
    def __init__(self, subsets, operator=torch.mul):
        """
        Parameters
        ----------
        subsets : list[tuple[int]]
            The subsets of vectors to compute the outer product. All subsets must be the same size.
            
        operator : Callable[[torch.Tensor, torch.Tensor], torch.Tensor], default=torch.mul
            This allow the user to specify another binary operator. For example, it can be helpful
            to perform the outer product in log space. In which case, operator can be set to torch.add.
            
        Notes
        -----
        This algorithm does not assume any proprety of the "operator". To compute the outer product of v, w, and z,
        it computes (v*w)*z. If another subset starts with (v, w) the computation will be shared. However, if another
        subset ends with (w, z) there will be no obtimization. By supposing associativity we might find a more
        efficient order (other than left to right) of "multiplication". By assuming commutativity, it would be possible
        to be even more efficient. Finding the optimal order of multiplication given the subsets seems quite hard.
        
        Anyway, the computation will be more efficient if many subsets share the same start.
        """
        subsets = list(subsets)
        min_k = min(map(len, subsets))
        max_k = max(map(len, subsets))
        if min_k != max_k:
            raise ValueError(f'All subsets must be the same size, found sizes ranging from {min_k} to {max_k}.')
        
        self.k = min_k
        self.subsets = subsets
        self.operator = operator
        
        
        head_tuples = unique_slice(subsets, start=0, end=2)
        self.head_indexes = {2: torch.tensor([s[0] for s in head_tuples])}
        self.tail_indexes = {2: torch.tensor([s[1] for s in head_tuples])}
        self.build_order = {2: head_tuples}

        for i in range(3, self.k+1):
            new_head_tuples = unique_slice(subsets, start=0, end=i)
            self.head_indexes[i] = torch.tensor(sub_index(new_head_tuples, head_tuples, start=0, end=i-1))
            self.tail_indexes[i] = torch.tensor([s[i-1] for s in new_head_tuples])
            self.build_order[i] = head_tuples = new_head_tuples
        
    def __call__(self, vectors):
        """
        Parameters
        ----------
        vectors : torch.tensor, vector.shape=(*sh, n, d)
        
        Returns
        -------
        outer : torch.tensor, vector.shape=(*sh, N, d**k)
            outer[..., i, j_1*d**(k-1) + j_2*d**(k-2) + ... + j_k] = vectors[..., a_1, j_1]*...*vectors[..., a_k, j_k]
            with (a_1, a_2, ..., a_k) = subsets[i] and N = len(subsets)
        """
        prev = vectors
        *sh, n, d = vectors.shape
        for i in range(2, self.k+1):
            head_index = self.head_indexes[i]
            tail_index = self.tail_indexes[i]
            prev = self.operator(prev[..., head_index, :, None], vectors[..., tail_index, None, :])
            prev = prev.view(*sh, len(self.build_order[i]), d**i)
        return prev

def iterslice(start=0, stop=None, slice_size=None):
    stop = float('inf') if stop is None else stop
    slice_size = stop-start if slice_size is None else slice_size
    if slice_size <= 0: raise ValueError('slice_size <= 0')
    
    while True:
        if stop <= start: return
        new_start = min(start + slice_size, stop)
        yield start, new_start
        start = new_start
        
def entropy_from_log_probabilities(log_p):
    # entropy taken along the last dim (i.e., p.exp().sum(dim=-1) must be 1)
    return -np.log2(np.e)*(log_p.exp()*log_p).sum(dim=-1)

def entropy_from_probabilities(p):
    # entropy taken along the last dim (i.e., p.sum(dim=-1) must be 1)
    p = torch.where(p==0, torch.ones_like(p), p) #convert p=0 to p=1 to avoid log(0)
    return -(p * torch.log2(p)).sum(dim=-1)
    
class BitsCombinations:
    def __init__(self, subsets):
        self.outer_product = PartialOuterProduct(subsets, operator=torch.logical_and)
        
    def log_distributions(self, codes):
        #this will returns many -inf
        return self.distributions(codes).log()
    
    def distributions(self, codes):
        #returns onehots
        base_case = torch.stack([~codes, codes], dim=-1)
        return self.outer_product(base_case)
    
    def log_average_distributions(self, codes, batch_size=None):
        #can return -inf
        return self.average_distributions(codes, batch_size=batch_size).log()
    
    def average_distributions(self, codes, batch_size=None):
        N = len(codes)
        for i, (beg,end) in enumerate(iterslice(stop=N, slice_size=batch_size)):
            batch_sum_dists = self.distributions(codes[beg:end]).float().sum(dim=0)
            sum_dists = batch_sum_dists if i==0 else sum_dists + batch_sum_dists
        return sum_dists / N
    
    def average_distributions_entropy(self, codes, batch_size=None):
        average_distributions = self.average_distributions(codes, batch_size=batch_size)
        return entropy_from_probabilities(average_distributions)
    

class BernoulliCombinations:
    def __init__(self, subsets):
        self.outer_product = PartialOuterProduct(subsets, operator=torch.add)
        
    def log_distributions(self, logits):
        #distribution (in log prob) of combinations of size 1 (base case) shape=(bs, n choose 1, 2**1) = (bs, n, 2)
        base_case = torch.stack([log_sigmoid(-logits), log_sigmoid(logits)], dim=-1)
        return self.outer_product(base_case)
    
    def distributions(self, logits):
        return self.log_distributions(logits).exp()
    
    def log_average_distributions(self, logits, batch_size=None):
        N = len(logits)
        for i, (beg,end) in enumerate(iterslice(stop=N, slice_size=batch_size)):
            batch_log_sum_dists = torch_lse(self.log_distributions(logits[beg:end]), dim=0)
            log_sum_dists = batch_log_sum_dists if i==0 else torch_logsumexp(log_sum_dists, batch_log_sum_dists)
        return log_sum_dists - np.log(N)
    
    def average_distributions(self, logits, batch_size=None):
        return self.average_log_distributions(logits, batch_size=batch_size).exp()
    
    def average_distributions_entropy(self, logits, batch_size=None):
        average_log_distributions = self.log_average_distributions(logits, batch_size=batch_size)
        return entropy_from_log_probabilities(average_log_distributions)