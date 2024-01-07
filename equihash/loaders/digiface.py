import numpy as np
import h5py, torch
from equihash.utils.unique_randint import unique_randint, unique_randint_mosaic
from .abstractloader import AbstractLoader, AbstractMosaicLoader

_digiface_splits = {
    'train': (0, 40_000),
    'valid': (40_000, 49_999),
    'test': (49_999, 109_999),
}

_digiface_mosaic_splits = {
    'train': (0, 36_000),
    'valid': (36_000, 72_000),
    'test': (72_000, 108_000),
}

class DigiFace(AbstractLoader):
    def __init__(self, path, size=None, instances=[0,1,2,3,4], which='train', split=None, dtype=torch.float32, device='cpu', seed=None):
        a, b = _digiface_splits[which] if split is None else split
        with h5py.File(path, 'r') as f: x = f['images'][a:b][:,instances]
        self._x = torch.tensor(x, dtype=torch.uint8, device=device)
        self.nb_instances_per_labels = len(instances)
        
        super().__init__(size=size, which=which, dtype=dtype, device=device, seed=seed)
        
    @property
    def x(self):
        return self._x
    
    def batch_from_labels(self, labels, nb_instances=None, torch_generator=None, numpy_generator=None):
        torch_generator = self.torch_generator if torch_generator is None else torch_generator
        numpy_generator = self.numpy_generator if numpy_generator is None else numpy_generator
        if nb_instances is None:
            instances = torch.randint(
                0, self.nb_instances_per_labels, labels.shape, generator=torch_generator, device=self.device)
        else:
            instances = unique_randint(
                0, self.nb_instances_per_labels, n=len(labels), k=nb_instances, rng=numpy_generator)
            instances = torch.tensor(instances, device=self.device)
            labels = torch.stack(nb_instances*[labels], dim=1)
        return self.x[labels, instances]
    
class DigiFaceMosaic(AbstractMosaicLoader):
    def __init__(self, path, height=1, width=2, size=None, instances=[0,1,2,3,4], which='train', split=None, dtype=torch.float32, device='cpu', seed=None):
        self.height = height
        self.width = width
        
        a, b = _digiface_splits[which] if split is None else split
        with h5py.File(path, 'r') as f: x = f['images'][a:b][:,instances]
        self._x = torch.tensor(x, dtype=torch.uint8, device=device)
        self.nb_instances_per_labels = len(instances)
        
        super().__init__(height=height, width=width, size=size, which=which, dtype=dtype, device=device, seed=seed)
        
    @property
    def x(self):
        return self._x
    
    def mosaic(self, labels, instances):
        *sh, h, w = labels.shape
        x = self.x[labels, instances]
        return x.view(-1, h, w, 112, 112, 3).permute(0, 1, 3, 2, 4, 5).reshape(*sh, h*112, w*112, 3)
    
    def batch_from_labels(self, labels, nb_instances=None, torch_generator=None, numpy_generator=None):
        torch_generator = self.torch_generator if torch_generator is None else torch_generator
        numpy_generator = self.numpy_generator if numpy_generator is None else numpy_generator
        k = 1 if nb_instances is None else nb_instances
        *sh, h, w = labels.shape
        labels = torch.stack(k*[labels], dim=-3)
        instances = unique_randint_mosaic(
            0, self.nb_instances_per_labels,
            n=np.prod(sh, dtype=int), k=k,
            mosaic_shape=self.mosaic_shape,
            rng=numpy_generator
        ).reshape(*sh, k, *self.mosaic_shape)
        instances = torch.tensor(instances)
        mosaic = self.mosaic(labels, instances)
        if nb_instances is None:
            return mosaic[...,0,:,:,:]
        return mosaic