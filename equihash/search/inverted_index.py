import torch, h5py
import numpy as np

from ._inverted_index import FNV1, cython_linear_build, cython_quadratic_build, cython_search

NULL = 2**32 - 1
_supported_modes = ('r', 'w', 'a')

def python_search(code, heads, links, codes):
    h = heads[FNV1(code)%(heads.shape[0])]
    index = []
    while h != NULL:
        if (code==codes[h]).all():
            index.append(h)
        h = links[h]
    return index

class InvertedIndex(object):
    def __init__(self, path, mode):
        if mode not in _supported_modes:
            raise ValueError(f'mode=\'{mode}\' is not supported, supported modes are: {_supported_modes}.')
        self.path = path
        self.mode = mode
        self.heads = None
        self.links = None
        self.codes = None
        self._file = None
        self._loaded = False
        
    @property
    def readonly(self):
        return self.mode=='r'
    
    @property
    def loaded(self):
        return self._loaded
    
    @property
    def closed(self):
        return not bool(self._file)
    
    @property
    def missing(self):
        return tuple(k for k in ('heads', 'links', 'codes') if getattr(self, k) is None)
    
    @property
    def linked(self):
        return not self.closed and not self.missing
    
    @property
    def ready(self):
        return self.loaded or self.linked
    
    def _check_ready(self):
        if not self.ready:
            raise ValueError('Index is not ready. Use the with statement or load the index.')
    
    @property
    def size(self):
        self._check_ready()
        return len(self.codes)
    
    @property
    def nb_heads(self):
        self._check_ready()
        return len(self.heads)
        
    def build(self, nb_heads, codes, algo='linear'):
        self.heads = np.zeros((nb_heads,), dtype=np.uint32)-1
        self.links = np.zeros((len(codes),), dtype=np.uint32)-1
        self.codes = codes
        if algo == 'linear':
            cython_linear_build(self.heads, self.links, self.codes)
        elif algo == 'quadratic':
            cython_quadratic_build(self.heads, self.links, self.codes)
        else: raise ValueError(f'algo must be "linear" or "quadratic", got {algo}')
        self._loaded = True
        
    def __getitem__(self, code):
        self._check_ready()
        if self.loaded:
            return cython_search(code, self.heads, self.links, self.codes)
        return python_search(code, self.heads, self.links, self.codes)
    
    def __enter__(self):
        if not self.loaded:
            self._file = h5py.File(self.path, self.mode)
            self.heads = self._file['heads'] if 'heads' in self._file else None
            self.links = self._file['links'] if 'links' in self._file else None
            self.codes = self._file['codes'] if 'codes' in self._file else None
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if not self.closed:
            self._file.close()
        
    def load(self):
        if self.loaded:
            return
            
        f = self._file if not self.closed else h5py.File(self.path, 'r')
        
        try:
            self.heads = f['heads'][:]
            self.links = f['links'][:]
            self.codes = f['codes'][:]
            self._loaded = True
        finally:
            if self.closed:
                f.close()
        return self
        
    def _save_array(self, f, name):
        array = getattr(self, name)
        if name in f: f[name][:] = array
        else: f.create_dataset(name, data=array)
    
    def save(self):
        if not self.loaded:
            raise ValueError('Index is not loaded. There is nothing to save.')
        
        if self.readonly:
            raise ValueError('Cannot save if mode is read only.')
        
        f = self._file if not self.closed else h5py.File(self.path, 'a')
        
        try:
            self._save_array(f, 'heads')
            self._save_array(f, 'links')
            self._save_array(f, 'codes')
        finally:
            if self.closed:
                f.close()
        return self