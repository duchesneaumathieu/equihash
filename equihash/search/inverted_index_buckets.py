import h5py
import numpy as np
from ._inverted_index import cython_build_buckets

_supported_modes = ('r', 'w', 'a')
class InvertedIndexBuckets:
    def __init__(self, path=None, mode='r'):
        if mode not in _supported_modes:
            raise ValueError(f'mode=\'{mode}\' is not supported, supported modes are: {_supported_modes}.')
        self.path = path
        self.mode = mode
        self.heads = None
        self.bucks = None
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
        return tuple(k for k in ('heads', 'bucks') if getattr(self, k) is None)
    
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
        return len(self.heads)
        
    def build(self, inverted_index):
        with inverted_index:
            size = inverted_index.size
            self.heads = np.zeros((size, 2), dtype=np.uint32)-1
            self.bucks = np.zeros((size,  ), dtype=np.uint32)-1
            cython_build_buckets(inverted_index.heads, inverted_index.links, inverted_index.codes, self.heads, self.bucks)
        self._loaded = True
        return self
        
    def __getitem__(self, i):
        self._check_ready()
        a, b = self.heads[i]
        return self.bucks[a:b]
    
    def _check_path(self):
        if self.path is None:
            raise ValueError('path is None, please init the object with a path.')
        
    def __enter__(self):
        if not self.loaded:
            self._check_path()
            self._file = h5py.File(self.path, self.mode)
            self.heads = self._file['heads'] if 'heads' in self._file else None
            self.links = self._file['bucks'] if 'bucks' in self._file else None
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if not self.closed:
            self._file.close()
        
    def load(self):
        if self.loaded:
            return
            
        self._check_path()
        f = self._file if not self.closed else h5py.File(self.path, 'r')
        
        try:
            self.heads = f['heads'][:]
            self.bucks = f['bucks'][:]
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
        
        self._check_path()
        f = self._file if not self.closed else h5py.File(self.path, 'a')
        
        try:
            self._save_array(f, 'heads')
            self._save_array(f, 'bucks')
        finally:
            if self.closed:
                f.close()
        return self
