import torch, pickle, gzip
from .abstractloader import AbstractMosaicLoader

class MnistMosaic(AbstractMosaicLoader):
    def __init__(self, path, size=None, height=2, width=2, sigma=0.2, which='train', device='cpu', seed=None):
        self.sigma = sigma
        
        with gzip.open(path, 'rb') as f:
            x, y = pickle.load(f, encoding='latin1')[{'train':0, 'valid':1, 'test':2}[which]]
        self._x = torch.tensor(x, device=device)
        
        super().__init__(height, width, size=size, which=which, device=device, seed=seed)
        
    @property
    def x(self):
        return self._x
        
    def mosaic(self, index):
        *sh, h, w = index.shape
        x = self.x[index]
        return x.view(-1, h, w, 28, 28).permute(0, 1, 3, 2, 4).reshape(*sh, h*28, w*28)
    
    def batch_from_labels(self, index):
        mosaic = self.mosaic(index)
        noise = torch.normal(0, self.sigma, mosaic.shape, generator=self.torch_generator, device=self.device)
        return mosaic + noise