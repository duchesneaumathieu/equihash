import numpy as np

def enough_int(low, high, k):
    if high-low < k:
        msg = 'k cannot be bigger than high-low, got {} and {}.'
        raise ValueError(msg.format(k, high-low))
        
def isrepeat(x):
    """
    Parameters
    ----------
    x : numpy.ndarray (shape: (..., k))

    Returns
    -------
    z : numpy.ndarray (shape: x.shape[:-1], dtype: bool)
        z[i] is True iif x[i]'s elements are not unique
    """
    if x.ndim > 1:
        n, k = x.shape
        return np.array([isrepeat(r) for r in x])
    u, c = np.unique(x, return_counts=True)
    return (c!=1).any()

def unique_randint_with_permutation(low, high, n, k, dtype=np.int64, rng=np.random):
    """
    Similar to numpy.random.randint but where each rows have unique elements.
    
    Parameters
    ----------
    low : int
        Lowest (signed) integer to be drawn from the distribution.
    high : int
        Largest (signed) integer to be drawn from the distribution.
    n : int
        The (unsigned) number of rows (vectors) with unique elements.
    k : int 
        The (unsigned) size of each rows (vectors).
    dtype: type (optional)
        The output dtype. (default: np.int64)
    rng : numpy.random.generator.Generator (optional)
        The generator used for sampling. (default: np.random)

    Returns
    -------
    samples : np.ndarray (shape: (n, k), dtype: int)
        For all i<n, samples[i,j] == samples[i,k] iif j==k.
        
    Notes
    -----
    This implementation uses rng.permutation. The memory and time usage are in O(n*k + r) and O(r)
    respectively, with r = high-low.
    """
    enough_int(low, high, k)
    return np.array([(rng.permutation(high-low)+low)[:k] for i in range(n)], dtype=dtype)

def unique_randint_with_randint(low, high, n, k, dtype=np.int64, rng=np.random):
    """
    Similar to numpy.random.randint but where each rows have unique elements.
    
    Parameters
    ----------
    low : int
        Lowest (signed) integer to be drawn from the distribution.
    high : int
        Largest (signed) integer to be drawn from the distribution.
    n : int
        The (unsigned) number of rows (vectors) with unique elements.
    k : int 
        The (unsigned) size of each rows (vectors).
    dtype: type (optional)
        The output dtype. (default: np.int64)
    rng : numpy.random.generator.Generator (optional)
        The generator used for sampling. (default: np.random)

    Returns
    -------
    samples : np.ndarray (shape: (n, k), dtype: int)
        For all i<n, samples[i,j] == samples[i,k] iif j==k.
        
    Notes
    -----
    This implementation uses rng.randint with rejection sampling (i.e. reject when samples are not unique). This runs
    in O(n*k) memory and expected O(n*k) if k << high-low. This is way faster than shuffle/permutation if 
    k << high-low because the probability of accepting is close to one. This probability can be computed using
    p = (np.arange(r-k+1, r+1)/r).prod() with r = high-low.
    
    This implementation is recursive, consequently if p is to low (e.g. k is near high-low) then this will raise a
    RecursionError.
    """
    enough_int(low, high, k)
    samples = rng.randint(low, high, (n, k), dtype=dtype)
    not_valid = isrepeat(samples)
    if any(not_valid):
        samples[not_valid] = unique_randint_with_randint(low, high, sum(not_valid), k, dtype=dtype, rng=rng)
    return samples

class PartialPermutation(object):
    def __init__(self):
        self.table = dict()
    
    def __getitem__(self, i):
        return self.table[i] if i in self.table else i
    
    def switch(self, i, j):
        v = self[i]
        self.table[i] = self[j]
        self.table[j] = v

def fast_unique_randint(low, high, n, k, dtype=np.int64, rng=np.random):
    """
    Similar to numpy.random.randint but where each rows have unique elements.
    
    Parameters
    ----------
    low : int
        Lowest (signed) integer to be drawn from the distribution.
    high : int
        Largest (signed) integer to be drawn from the distribution.
    n : int
        The (unsigned) number of rows (vectors) with unique elements.
    k : int 
        The (unsigned) size of each rows (vectors).
    dtype: type (optional)
        The output dtype. (default: np.int64)
    rng : numpy.random.generator.Generator (optional)
        The generator used for sampling. (default: np.random)

    Returns
    -------
    samples : np.ndarray (shape: (n, k), dtype: int)
        For all i<n, samples[i,j] == samples[i,k] iif j==k.
        
    Notes
    -----
    The fastest unique_randint algorithm. Runs in O(n*k) memory and O(n*k) times. However this python
    implementation (without any vectorization) is quite slow.
    """
    enough_int(low, high, k)
    samples = np.zeros((n, k), dtype=dtype)
    for i in range(n):
        p = PartialPermutation()
        for j in range(k):
            r = rng.randint(j, high-low)
            samples[i, j] = p[r]+low
            p.switch(j, r)
    return samples

def unique_randint(low, high, n, k, dtype=np.int64, rng=np.random):
    """
    Similar to numpy.random.randint but where each rows have unique elements.
    
    Parameters
    ----------
    low : int
        Lowest (signed) integer to be drawn from the distribution.
    high : int
        Largest (signed) integer to be drawn from the distribution.
    n : int
        The (unsigned) number of rows (vectors) with unique elements.
    k : int 
        The (unsigned) size of each rows (vectors).
    dtype: type (optional)
        The output dtype. (default: np.int64)
    rng : numpy.random.generator.Generator (optional)
        The generator used for sampling. (default: np.random)

    Returns
    -------
    samples : np.ndarray (shape: (n, k), dtype: int)
        For all i<n, samples[i,j] == samples[i,k] iif j==k.
        
    Notes
    -----
    When the acceptance probability is fine, unique_randint_with_randint is used.
    Otherwise, if high-low < 500K then unique_randint_with_permutation is used. Otherwise, 
    fast_unique_randint is used.
    """
    enough_int(low, high, k)
    r = high-low
    #p = 1 * (r-1)/r * (r-2)/r * ... * (r-k+1)/r
    #log(p) = log(r-1) + log(r-2) + log(r-k+1) - k*log(r)
    log_p = (np.log(np.arange(r-k+1, r)) - np.log(r)).sum()
    if log_p > np.log(0.5):
        return unique_randint_with_randint(low, high, n, k, dtype=dtype, rng=rng)
    elif r < 500_000: #random rule of thumb
        return unique_randint_with_permutation(low, high, n, k, dtype=dtype, rng=rng)
    else:
        return fast_unique_randint(low, high, n, k, dtype=dtype, rng=rng)