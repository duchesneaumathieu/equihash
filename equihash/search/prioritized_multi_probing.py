import numpy as np
from itertools import islice
from heapq import heappush, heappop

def push_children(heap, subset, subsum, values, produced):
    if not subset or subset[0] != 0: #add the smallest index, e.g., (2,5,6,8) -> (0,2,5,6,8)
        new = (0,) + subset
        if new not in produced:
            produced.add(new)
            heappush(heap, (values[0] + subsum, new))
    for n, i in enumerate(subset, 1): #produce all shifts, e.g., (2,5,6,8) -> (3,5,6,8), (2,5,7,8), and (2,5,6,9)
        j = i + 1
        if (n < len(subset) and j < subset[n]) or (n == len(subset) and j < len(values)):
            new = subset[:n-1] + (j,) + subset[n:]
            if new not in produced:
                produced.add(new)
                heappush(heap, (subsum - values[i] + values[j], new))

def smallest_subset_sums(values, yield_sums=False):
    """
    Generator that yields the subset's index for a set of values in
    increasing order of their sum. The values must be all positive.
    
    Parameters
    ----------
    values : numpy.ndarray (ndim: 1)
        The set values from which to take the subsets
        
    Yields
    ------
    subset : list of int
        Subset of index of the values in increasing order of their sum.
    """
    if (values<0).any():
        raise ValueError('values must be positive')
        
    #handling unsorted values
    perm = np.argsort(values)
    values = values[perm] #sort them first
        
    heap = [(0, ())]
    produced = set()
    while heap:
        subsum, subset = heappop(heap)
        yield [perm[i] for i in subset] #unsort them after and make it a list.
        push_children(heap, subset, subsum, values, produced)

def prioritized_multi_probing(logits):
    code = 0 < logits
    for subset in smallest_subset_sums(np.abs(logits)):
        c = code.copy()
        c[subset] ^= True
        yield c

class PrioritizedMultiProbing:
    def __init__(self, inverted_index, number_lookups=1):
        self.inverted_index = inverted_index
        self.number_lookups = number_lookups
        self.bool_to_uint8 = np.arange(256, dtype=np.uint8).reshape(2,2,2,2,2,2,2,2)
    
    def __call__(self, logits):
        items = list()
        with self.inverted_index:
            for code in islice(prioritized_multi_probing(logits.cpu().numpy()), self.number_lookups):
                fingerprint = self.bool_to_uint8[tuple(code.astype(int).reshape(8,8).transpose())]
                items.extend(self.inverted_index[fingerprint])
        return items