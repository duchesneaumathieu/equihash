import torch
import numpy as np
from abc import ABC, abstractmethod

class ProceduralDB(ABC):
    def __init__(self, seed, size, chunk_size, device='cpu'):
        #Emulate a big database without the memory usage: db[i] is always db[i]
        #but db[i] is only created when asked.
        #This deterministically seed a RNG for every chunk it creates. 
        assert 0 <= seed < 0xffff #because seed is equivalent to seed + 0xffff*i for all i
        self.seed = seed
        self.size = size
        self.chunk_size = chunk_size
        self.torch_generator = torch.Generator(device=device)
        self.numpy_generator = np.random.default_rng()
        
    @abstractmethod
    def generate_slice(self, beg, end, torch_generator, numpy_generator):
        pass
    
    def __len__(self):
        return self.size
    
    def get_chunk_generator(self, chunk_id):
        seed = 0xffff*chunk_id + self.seed
        self.numpy_generator = np.random.default_rng(seed)
        self.torch_generator.manual_seed(seed)
        return self.torch_generator, self.numpy_generator
    
    def generate_chunk(self, chunk_id):
        beg = chunk_id * self.chunk_size
        end = beg + self.chunk_size
        torch_generator, numpy_generator = self.get_chunk_generator(chunk_id)
        return self.generate_slice(beg, end, torch_generator, numpy_generator)
    
    def positive_index(self, index):
        if not (-self.size <= index < self.size):
            raise IndexError(f'index {index} is out of bounds with size {self.size}')
        return index % self.size
    
    def _getitem_int(self, index):
        index = self.positive_index(index)
        chunk_id = index//self.chunk_size
        return self.generate_chunk(chunk_id)[index%self.chunk_size]
    
    def _getitem_list(self, indexes):
        indexes = [self.positive_index(k) for k in indexes]
        chunk_ids = [k//self.chunk_size for k in indexes]
        chunk_map = {i: self.generate_chunk(i) for i in set(chunk_ids)}
        
        chunk_indexes = [k%self.chunk_size for k in indexes]
        return torch.stack([chunk_map[i][j] for i, j in zip(chunk_ids, chunk_indexes)])
        
    def _getitem_tuple(self, indexes):
        return self._getitem_list(indexes)
    
    def _slice_boundary(self, b):
        b = max(b+self.size, 0) if b < 0 else b
        return min(b, self.size)
    
    def _getitem_slice(self, s):
        if s.step is not None and s.step == 0:
            raise ValueError('slice step cannot be zero')
        if s.step is not None and s.step < 0:
            raise NotImplementedError('negative step is not implemented')
        beg = 0 if s.start is None else self._slice_boundary(s.start)
        end = self.size if s.stop is None else self._slice_boundary(s.stop)
        if end <= beg:
            raise RuntimeError('empty slice')
            
        chunk_ids = sorted(set([k//self.chunk_size for k in range(beg, end)]))
        chunks = torch.cat([self.generate_chunk(i) for i in set(chunk_ids)])
        
        chunks_beg = beg % self.chunk_size
        chunks_end = end - (beg - chunks_beg)
        return chunks[chunks_beg:chunks_end:s.step]
        
    def __getitem__(self, k):
        k_type = type(k).__name__
        if not isinstance(k, (int, list, tuple, slice)):
            raise TypeError(f'indices must be integers, list, tuple, or slices. not {k_type}')
        return getattr(self, f'_getitem_{k_type}')(k)
    
class ProceduralLabelsDB(ProceduralDB):
    def __init__(self, labels, loader, seed):
        if not (0 <= seed < 0xffff//3):
            raise ValueError(f'the seed must be between 0 and {0xffff//3}')
        seed = 3*seed
        if loader.which == 'valid': seed += 1
        elif loader.which == 'test': seed += 2
        
        super().__init__(seed, len(labels), chunk_size=500, device=loader.device)
        self.labels = labels
        self.loader = loader
            
    def generate_slice(self, beg, end, torch_generator, numpy_generator):
        labels = self.labels[beg:end]
        labels = torch.tensor(labels.astype(int))
        return self.loader.batch_from_labels(labels, torch_generator=torch_generator, numpy_generator=numpy_generator)