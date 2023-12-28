import math, torch
import numpy as np
from datetime import datetime
from itertools import combinations

def timestamp(s, strf='%Y-%m-%d %H:%M:%S'):
    return f'[{datetime.now().strftime(strf)}] {s}'

def iterslice(start=0, stop=None, slice_size=None):
    stop = float('inf') if stop is None else stop
    slice_size = stop-start if slice_size is None else slice_size
    if slice_size <= 0: raise ValueError('slice_size <= 0')
    
    while True:
        if stop <= start: return
        new_start = min(start + slice_size, stop)
        yield start, new_start
        start = new_start
        
def covering_random_combinations(n, k, cover_k, rng=np.random):
    combs = list()
    seen_combs = set()
    for sub_comb in combinations(range(n), cover_k):
        while True: #rejection sampling
            comb = sub_comb + tuple(rng.randint(0, n, k-cover_k))
            comb = tuple(sorted(comb))
            if len(set(comb))==k and comb not in seen_combs:
                combs.append(comb)
                seen_combs.add(comb)
                break
    return sorted(combs)

def get_significand(v, base=10):
    p = 1
    while not (v%(base**(p+1))==v and v%(base**p)!=v):
        if v%(base**p)!=v: p += 1
        else: p -= 1
    return p

def ceil(x, sig=0, base=10):
    return math.ceil(x / base**sig) * base**sig

def error_format2(v, err, max_precision=None):
    err = ceil(err, get_significand(err))
    sig = get_significand(err)
    p = -sig if max_precision is None else min(max_precision, -sig)
    p = max(0, p)
    v_format = '{0:.{1}f}'.format(round(v, p), p)
    err_format = '{0:.{1}f}'.format(ceil(err, p), p)
    return v_format, err_format

def error_format(v, err, max_precision=None):
    err = ceil(err, get_significand(err))
    sig = get_significand(err)
    p = max(0, -sig) if max_precision is None else max(max_precision, -sig)
    v_format = '{0:.{1}f}'.format(round(v,-sig), p)
    err_format = '{0:.{1}f}'.format(err, p)
    return v_format, err_format
                    
def avg_integer_bootstrap(samples, nb_bootstraps=1000, batch_size=None, z=0.05, rng=np.random):
    batch_size = nb_bootstraps if batch_size is None else batch_size
    if nb_bootstraps%batch_size != 0:
        raise NotImplementedError(f'nb_bootstraps ({nb_bootstraps}) must be a multiple of batch_size ({batch_size})')
    s, c = np.unique(samples, return_counts=True)
    avgs = list()
    for _ in range(nb_bootstraps // batch_size):
        bootstraps = rng.multinomial(len(samples), c/c.sum(), size=batch_size)
        avgs.append( (bootstraps * s).sum(axis=1) / len(samples) )
    avgs = np.concatenate(avgs)
    return np.quantile(avgs, [z/2, 1-z/2])

def avg_integer_bootstrap_format(samples, nb_bootstraps=1000, batch_size=None, z=0.05, max_precision=None, rng=np.random):
    avg = samples.mean()
    lower_bound, upper_bound = avg_integer_bootstrap(samples, nb_bootstraps=nb_bootstraps, batch_size=batch_size, z=z, rng=rng)
    err = max(abs(avg-lower_bound), abs(avg-upper_bound))
    return error_format(avg, err, max_precision=max_precision)

class PolynomialStepScheduler:
    def __init__(self, init, gamma, power, step_size):
        self.init = init
        self.gamma = gamma
        self.power = power
        self.step_size = step_size
        
    def __call__(self, step):
        truncated_step = step//self.step_size * self.step_size
        return self.init * math.pow(1.+self.gamma*truncated_step, self.power)

class ValueGradClipper:
    def __init__(self, params, clip_value):
        self.params = list(params)
        self.clip_value = clip_value
        self.nb_params = 0
        for p in self.params:
            self.nb_params += np.prod(p.shape)
        
    def clip_ratio(self):
        cumul = 0
        for p in self.params:
            cumul += (p.grad.abs()>self.clip_value).float().sum().tolist()
        return cumul / self.nb_params
    
    def clip(self):
        clip_ratio = self.clip_ratio()
        torch.nn.utils.clip_grad_value_(self.params, self.clip_value)
        return clip_ratio

def difference(x, y):
    if type(x) != type(y):
        return f'{type(x).__name__}-->{type(y).__name__}'
    if isinstance(x, dict):
        return dict_diff(x, y)
    if x != y:
        return f'{x}-->{y}'
    return ''

class Empty: pass
def dict_diff(d1, d2):
    all_keys = list(d1.keys()) + list(d2.keys())
    d1 = {k:d1[k] if k in d1 else Empty() for k in all_keys}
    d2 = {k:d2[k] if k in d2 else Empty() for k in all_keys}
    diff = {k:difference(d1[k], d2[k]) for k in all_keys}
    return {k:v for k,v in diff.items() if v}