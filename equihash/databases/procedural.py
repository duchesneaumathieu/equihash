import torch
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
        self.generator = torch.Generator(device=device)
        
    @abstractmethod
    def generate_slice(self, beg, end, generator):
        pass
    
    def __len__(self):
        return self.size
    
    def get_chunk_generator(self, chunk_id):
        seed = 0xffff*chunk_id + self.seed
        return self.generator.manual_seed(seed)
    
    def generate_chunk(self, chunk_id):
        beg = chunk_id * self.chunk_size
        end = beg + self.chunk_size
        generator = self.get_chunk_generator(chunk_id)
        return self.generate_slice(beg, end, generator)
    
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
    
class ProceduralMosaicDB(ProceduralDB):
    def __init__(self, mosaic_index, mosaic_loader, seed, chunk_size):
        size, mosaic_size = mosaic_index.shape
        h, w = mosaic_loader.height, mosaic_loader.width
        if mosaic_size != h*w:
            msg = f'mosaic_index has {mosaic_size} index which does not correspond to the loader mosaic shape ({h}x{w})'
            raise ValueError(msg)
            
        if not (0 <= seed < 0xffff//3):
            raise ValueError(f'the seed must be between 0 and {0xffff//3}')
        seed = 3*seed + mosaic_loader.which_code
        
        super().__init__(seed, size, chunk_size, device=mosaic_loader.device)
        self.mosaic_index = mosaic_index.reshape(size, h, w)
        self.mosaic_loader = mosaic_loader
            
    def generate_slice(self, beg, end, generator):
        mosaic_index = self.mosaic_index[beg:end]
        mosaic_index = torch.tensor(mosaic_index.astype(int))
        return self.mosaic_loader.noisy_mosaic(mosaic_index, generator)