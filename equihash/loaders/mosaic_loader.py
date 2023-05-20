import torch
from abc import ABC, abstractmethod

class MosaicLoader(ABC):
    def __init__(self, height, width, size=None, which='train', device='cpu', seed=None):
        self.height = height
        self.width = width
        self.size = size
        self.which = which
        self.device = device
        self.seed = seed
        self.generator = torch.Generator(device=device)
        if seed is not None:
            self.generator.manual_seed(seed)
        
        self.which_code = {'train':0, 'valid':1, 'test':2}[which]
            
    @property
    @abstractmethod
    def x(self):
        pass
    
    @abstractmethod
    def noisy_mosaic(self, index, generator):
        pass
    
    def negative_batch(self, size=None, generator=None):
        size = self._parse_size(size)
        generator = self.generator if generator is None else generator
        index = self._generate_negative_index(size, generator)
        return self.noisy_mosaic(index, generator)

    def positive_batch(self, size=None, generator=None):
        size = self._parse_size(size)
        generator = self.generator if generator is None else generator
        index = self._generate_positive_index(size, generator)
        return self.noisy_mosaic(index, generator)
    
    def batch(self, size=None, generator=None):
        size = self._parse_size(size)
        generator = self.generator if generator is None else generator
        nb_pos = size//2
        s = torch.zeros(size, dtype=bool, device=self.device); s[:nb_pos] = True
        positive_batch = self.positive_batch(size=nb_pos, generator=generator)
        negative_batch = self.negative_batch(size=size-nb_pos, generator=generator)
        return torch.cat([positive_batch, negative_batch], dim=0), s
    
    def _parse_size(self, size):
        size = self.size if size is None else size
        if size is None:
            raise ValueError('size must be set at instantiation or when calling the method.')
        return size
        
    def _generate_positive_index(self, size, generator):
        shape = (size, 1, self.height, self.width)
        index = torch.randint(0, len(self.x), shape, device=self.device, generator=generator)
        return torch.cat([index, index], dim=1)
    
    def _generate_negative_index(self, size, generator):
        shape = (size, 2, self.height*self.width)
        fail = True
        while fail: #do while loop
            index = torch.randint(0, len(self.x), shape, device=self.device, generator=generator)
            fail = (index[:,0]==index[:,1]).all(dim=1).any(dim=0)
        return index.view(size, 2, self.height, self.width)
    
    def get_state(self):
        return self.generator.get_state()
    
    def set_state(self, state):
        self.generator.set_state(state)
        return self