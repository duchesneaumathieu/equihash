import h5py, torch
from .abstractloader import AbstractMosaicLoader

_mnist_mosaic_splits = {
    'train': (0, 50_000),
    'valid': (50_000, 60_000),
    'test': (60_000, 70_000),
}

class MnistMosaic(AbstractMosaicLoader):
    def __init__(self, path, size=None, height=2, width=2, sigma=0.2, which='train', split=None, dtype=torch.float32, device='cpu', seed=None):
        self.sigma = sigma
        self.height = height
        self.width = width
        
        a, b = _mnist_mosaic_splits[which] if split is None else split
        with h5py.File(path, 'r') as f: x = f['x'][a:b]
        self._x = torch.tensor(x, device=device).type(dtype)/255
        
        super().__init__(height=height, width=width, size=size, which=which, dtype=dtype, device=device, seed=seed)
        
    @property
    def x(self):
        return self._x
        
    def mosaic(self, index):
        *sh, h, w = index.shape
        x = self.x[index]
        return x.view(-1, h, w, 28, 28).permute(0, 1, 3, 2, 4).reshape(*sh, h*28, w*28)
    
    def batch_from_labels(self, index, nb_instances=None, torch_generator=None, numpy_generator=None):
        torch_generator = self.torch_generator if torch_generator is None else torch_generator
        numpy_generator = self.numpy_generator if numpy_generator is None else numpy_generator
        if nb_instances is not None:
            index = torch.stack(nb_instances*[index], dim=1)
        mosaic = self.mosaic(index)
        noise = torch.normal(0, self.sigma, mosaic.shape, generator=torch_generator, device=self.device)
        return mosaic + noise