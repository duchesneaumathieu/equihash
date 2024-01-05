import torch
import numpy as np
from abc import ABC, abstractmethod
from equihash.utils.unique_randint import unique_randint, unique_randint_mosaic

class AbstractLoader(ABC):
    def __init__(self, size=None, which='train', dtype=torch.float32, device='cpu', seed=None):
        self.size = size
        self.which = which
        self.dtype = dtype
        self.device = device
        self.seed = seed
        self.torch_generator = torch.Generator(device=device)
        if seed is not None:
            self.numpy_generator = np.random.RandomState(seed)
            self.torch_generator.manual_seed(seed)
        else: self.numpy_generator = np.random
        
    def generate_labels(self, n, k):
        return unique_randint(0, len(self.x), n=n, k=k, dtype=np.int64, rng=self.numpy_generator)
    
    @property
    @abstractmethod
    def x(self):
        pass
    
    @abstractmethod
    def batch_from_labels(self, labels, nb_instances=None):
        pass
    
    def _parse_size(self, size):
        size = self.size if size is None else size
        if size is None:
            raise ValueError('size must be set at instantiation or when calling the method.')
        return size
    
    def manual_seed(self, seed):
        self.seed = seed
        self.numpy_generator.seed(seed)
        self.torch_generator.manual_seed(seed)
        return self
    
    def generate_negative_labels(self, size, k, replacement=True):
        if replacement:
            labels = self.generate_labels(n=size, k=k)
        else:
            labels = self.generate_labels(n=1, k=size*k)
            labels = labels.reshape(size, k, *labels.shape[2:])
        return torch.tensor(labels, device=self.device)
    
    def labels_batch(self, size=None, replacement=True):
        size = self._parse_size(size)
        if replacement: labels = self.generate_labels(n=size, k=1)[:,0]
        else: labels = self.generate_labels(n=1, k=size)[0]
        labels = torch.tensor(labels, device=self.device)
        mosaics = self.batch_from_labels(labels)
        return mosaics, labels
    
    def negative_pairs_batch(self, size=None, replacement=True):
        size = self._parse_size(size)
        labels = self.generate_negative_labels(size, 2, replacement=replacement)
        return self.batch_from_labels(labels)
    
    def positive_pairs_batch(self, size=None, replacement=True):
        size = self._parse_size(size)
        if replacement: labels = self.generate_labels(n=size, k=1)[:,0]
        else: labels = self.generate_labels(n=1, k=size)[0]
        labels = torch.tensor(labels, device=self.device)
        return self.batch_from_labels(labels, nb_instances=2)
    
    def mixed_pairs_batch(self, size=None, nb_positive=None, nb_negative=None, replacement=True):
        if nb_positive is not None and nb_negative is not None:
            computed_size = nb_positive + nb_negative
        else: computed_size = self._parse_size(size)
        if size is not None and computed_size != size:
            msg = ('Incompatible set of paramters:'
                   ' nb_positive + nb_negative = size but the method received'
                   f' {nb_positive} + {nb_negative} != {size}.')
            raise ValueError(msg)
        size = computed_size
        if nb_positive is None:
            if nb_negative is None: nb_positive = size // 2
            elif size < nb_negative:
                raise ValueError(f'size(={size}) < nb_negative(={nb_negative}).')
            else: nb_positive = size - nb_negative
        if size < nb_positive:
            raise ValueError(f'size(={size}) < nb_positive(={nb_positive}).')
        nb_negative = size - nb_positive
        s = torch.zeros(size, dtype=bool, device=self.device); s[:nb_positive] = True
        labels = self.generate_negative_labels(size, 2, replacement=replacement)
        positive_instances = self.batch_from_labels(labels[:nb_positive, 0], nb_instances=2)
        negative_instances = self.batch_from_labels(labels[nb_positive:])
        batch = torch.cat([positive_instances, negative_instances], dim=0)
        return batch, s
    
    def hinge_triplets_batch_from_negative_labels(self, labels):
        positive_instances = self.batch_from_labels(labels[:, 0], nb_instances=2)
        negative_instances = self.batch_from_labels(labels[:, 1], nb_instances=1)
        return torch.cat([positive_instances, negative_instances], dim=1)
    
    def hinge_triplets_batch(self, size=None, replacement=True):
        size = self._parse_size(size)
        labels = self.generate_negative_labels(size, 2, replacement=replacement)
        return self.hinge_triplets_batch_from_negative_labels(labels)
    
    def get_state(self):
        return (self.numpy_generator.get_state(), self.torch_generator.get_state())
    
    def set_state(self, state):
        numpy_state, torch_state = state
        self.numpy_generator.set_state(numpy_state)
        self.torch_generator.set_state(torch_state)
        return self

class AbstractMosaicLoader(AbstractLoader):
    def __init__(self, height, width, size=None, which='train', dtype=torch.float32, device='cpu', seed=None):
        self.height = height
        self.width = width
        self.mosaic_shape = (height, width)
        super().__init__(size=size, which=which, dtype=dtype, device=device, seed=seed)
        
    def generate_labels(self, n, k):
        return unique_randint_mosaic(
            0, len(self.x), n=n, k=k, mosaic_shape=self.mosaic_shape, dtype=np.int64, rng=self.numpy_generator)