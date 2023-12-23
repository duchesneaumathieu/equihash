import torch
import numpy as np
from abc import ABC, abstractmethod
from equihash.utils.labels_generator import LabelsGenerator, MosaicLabelsGenerator

class BaseAbstractLoader(ABC):
    def __init__(self, size=None, which='train', device='cpu', seed=None):
        self.size = size
        self.which = which
        self.device = device
        self.seed = seed
        self.torch_generator = torch.Generator(device=device)
        if seed is not None:
            self.numpy_generator = np.random.RandomState(seed)
            self.torch_generator.manual_seed(seed)
        else: self.numpy_generator = np.random
    
    @property
    @abstractmethod
    def x(self):
        pass
    
    @abstractmethod
    def batch_from_labels(self, labels):
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
    
    def labels_batch(self, size=None, replacement=True):
        size = self._parse_size(size)
        labels = self.labels_generator.generate_labels(size, replacement=replacement)
        labels = torch.tensor(labels, device=self.device)
        mosaics = self.batch_from_labels(labels)
        return mosaics, labels
    
    def negative_pairs_batch(self, size=None, replacement=True):
        size = self._parse_size(size)
        labels = self.labels_generator.generate_negative_pairs(size, replacement=replacement)
        labels = torch.tensor(labels, device=self.device)
        return self.batch_from_labels(labels)
    
    def positive_pairs_batch(self, size=None, replacement=True):
        size = self._parse_size(size)
        labels = self.labels_generator.generate_positive_pairs(size, replacement=replacement)
        labels = torch.tensor(labels, device=self.device)
        return self.batch_from_labels(labels)
    
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
        labels = self.labels_generator.generate_mixed_pairs(nb_positive, nb_negative, replacement=replacement)
        labels = torch.tensor(labels, device=self.device)
        return self.batch_from_labels(labels), s
    
    def hinge_triplets_batch(self, size=None, replacement=True):
        size = self._parse_size(size)
        labels = self.labels_generator.generate_hinge_triplet(size, replacement=replacement)
        labels = torch.tensor(labels, device=self.device)
        return self.batch_from_labels(labels)
    
    def get_state(self):
        return (self.numpy_generator.get_state(), self.torch_generator.get_state())
    
    def set_state(self, state):
        numpy_state, torch_state = state
        self.numpy_generator.set_state(numpy_state)
        self.torch_generator.set_state(torch_state)
        return self

class AbstractLoader(BaseAbstractLoader):
    def __init__(self, size=None, which='train', device='cpu', seed=None):
        super().__init__(size=size, which=which, device=device, seed=seed)
        self.labels_generator = LabelsGenerator(
            len(self.x),
            replacement=True,
            dtype=np.int64,
            rng=self.numpy_generator
        )
    
class AbstractMosaicLoader(BaseAbstractLoader):
    def __init__(self, height, width, size=None, which='train', device='cpu', seed=None):
        super().__init__(size=size, which=which, device=device, seed=seed)
        self.height = height
        self.width = width
        self.mosaic_shape = (height, width)
        self.labels_generator = MosaicLabelsGenerator(
            len(self.x),
            mosaic_shape=self.mosaic_shape,
            replacement=True,
            dtype=np.int64,
            rng=self.numpy_generator
        )