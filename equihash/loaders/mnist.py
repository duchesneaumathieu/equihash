import torch, pickle, gzip
from .mosaic_loader import MosaicLoader

class MnistMosaic(MosaicLoader):
    def __init__(self, path, size=None, height=2, width=2, sigma=0.2, which='train', device='cpu', seed=None):
        super().__init__(height, width, size=size, which=which, device=device, seed=seed)
        self.sigma = sigma
        
        with gzip.open(path, 'rb') as f:
            x, y = pickle.load(f, encoding='latin1')[self.which_code]
        self._x = torch.tensor(x, device=device)
        
    @property
    def x(self):
        return self._x
        
    def mosaic(self, index):
        *sh, h, w = index.shape
        x = self.x[index]
        return x.view(-1, h, w, 28, 28).permute(0, 1, 3, 2, 4).reshape(*sh, h*28, w*28)
    
    def noisy_mosaic(self, index, generator):
        mosaic = self.mosaic(index)
        noise = torch.normal(0, self.sigma, mosaic.shape, generator=generator, device=self.device)
        return mosaic + noise