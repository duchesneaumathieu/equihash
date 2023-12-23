import torch, h5py
import numpy as np
from .abstractloader import AbstractMosaicLoader

def torch_uniform(minimum, maximum, shape, *args, **kwargs):
    delta = maximum - minimum
    return torch.rand(shape, *args, **kwargs)*delta + minimum

class Overlap2x1Mosaic(object):
    def __init__(self, image_size, mosaic_size, p=0.99, device='cuda', dtype=torch.float32):
        self.image_size = image_size
        self.mosaic_size = mosaic_size
        self.overlap = 2*image_size - mosaic_size
        
        logit = np.log(p) - np.log(1-p)
        linspace = torch.linspace(-logit, logit, self.overlap, device=device, dtype=dtype)
        self.right = torch.ones((image_size,), device=device, dtype=dtype)
        self.right[:self.overlap] = torch.sigmoid(linspace)
        self.left = torch.flip(self.right, dims=(0,))
        
    def __call__(self, batch):
        shape = (len(batch), 3, self.image_size, self.mosaic_size)
        im = torch.zeros(shape, device=batch.device, dtype=batch.dtype)
        im[..., :self.image_size] += self.left * batch[:,0]
        im[..., -self.image_size:] += self.right * batch[:,1]
        return im

class ImageFormatConverter(object):
    def __init__(self, device='cpu'):
        self.torch_hls_to_rgb_n = torch.tensor([[[0]], [[8]], [[4]]], dtype=torch.float32, device=device)
        
    def torch_rgb_to_hls(self, images):
        """
        Transform a batch of RGB images into HLS format.

        Parameters
        ----------
        images : torch.Tensor (shape (N, 3, H, W))
            The images to transform.

        Returns
        -------
        hls_images : torch.Tensor (shape (N, 3, H, W))
            Instead of red, green and blue there is hue, luminance and saturation.
        """
        #BS, C, H, W
        vmin, imin = images.min(dim=1)
        vmax, imax = images.max(dim=1)
        c = vmax - vmin
        zeros = c==0
        pc = c.masked_fill_(zeros, 1)

        #hue
        pim = images/pc[:,None]
        r, g, b = pim[:,0], pim[:,1], pim[:,2]
        ph = (imax==0)*(g-b) + (imax==1)*(b-r+2) + (imax==2)*(r-g+4)
        h = ph.masked_fill(zeros, 0)/6 % 1

        #luminance
        l = (vmax + vmin)/2

        #saturation
        pl = l.masked_fill(zeros, 1)
        ps = c / (1 - torch.abs(2*pl - 1))
        s = ps.masked_fill(zeros, 0)

        return torch.stack([h,l,s], dim=1)
    
    def torch_hls_to_rgb(self, hls_images):
        """
        Transform a batch of RGB images into HLS format.

        Parameters
        ----------
        hls_images : torch.Tensor (shape (N, 3, H, W))
            The images, where the channels correspond to hue, luminance and saturation.

        Returns
        -------
        images : torch.Tensor (shape (N, 3, H, W))
            Instead of hue, luminance and saturation there is red, green and blue.
        """
        im = hls_images
        h, l, s = im[:,0][:, None], im[:,1][:, None], im[:,2][:, None]
        a = s*torch.min(l, 1-l)
        h12 = 12*h
        k = (self.torch_hls_to_rgb_n+h12)%12
        v = torch.min(k-3, 9-k).clamp(-1, 1)
        return (l-a*v).clamp(0, 1)

def build_grid(H, W, device='cpu'):
    hlin = torch.linspace(-1, 1, H, device=device)
    wlin = torch.linspace(-1, 1, W, device=device)
    y, x = torch.meshgrid(hlin, wlin)
    return torch.stack([x, y], dim=2)

class ImageDistortion(object):
    def __init__(self, perlin_path, min=0, max=2, p=2, device='cpu', limit=None):
        with h5py.File(perlin_path, 'r') as f:
            self.perlin1 = torch.tensor(f['perlin1'][:limit], device=device)
            self.perlin2 = torch.tensor(f['perlin2'][:limit], device=device)
        self.shape = self.perlin1.shape[1:]
        self.grid = build_grid(*self.shape, device=device)
        self.circle = torch.sigmoid(-8*(self.grid**2).sum(dim=-1).sqrt() + 6)
        self.proot_min = min**(1/p)
        self.proot_max = max**(1/p)
        self.p = p
            
    def spacial_distortion(self, images, perlin):
        distorted_grid = self.grid + self.circle[...,None]*perlin
        return torch.nn.functional.grid_sample(images, distorted_grid, align_corners=False, padding_mode='border')
    
    def sample_perlin_noise(self, n, generator):
        n11 = self.perlin1[torch.randint(len(self.perlin1), (n,), device=generator.device, generator=generator)]
        n22 = self.perlin2[torch.randint(len(self.perlin2), (n,), device=generator.device, generator=generator)]
        return n11/256 + n22/512
    
    def __call__(self, images, generator):
        n, _, h, w = images.shape
        global_noise = torch_uniform(self.proot_min, self.proot_max, n, device=images.device, generator=generator)**self.p
        perlin = self.sample_perlin_noise(2*n, generator).view(n,2,h,w).permute(0,2,3,1)
        return self.spacial_distortion(images, global_noise.view(n,1,1,1)*perlin)

class NaturalMosaic(AbstractMosaicLoader):
    def __init__(self, images_path, perlin_path,
                 size=None, which='train', device='cpu', seed=None,
                 nb_images=2**16, perlin_limit=None):
        super().__init__(1, 2, size=size, which=which, device=device, seed=seed)
        
        with h5py.File(images_path, 'r') as f:
            beg = 2**16 * self.which_code
            imgs = f['dataset'][beg:beg+nb_images]
            self._x = torch.tensor(imgs, device=device, dtype=torch.uint8).permute(0, 3, 1, 2)
            
        with h5py.File(perlin_path, 'r') as f:
            mosaic_size = f['perlin1'].shape[-1]
            
        n, c, h, w = self.x.shape
        self.mosaic_builder = Overlap2x1Mosaic(image_size=h, mosaic_size=mosaic_size, p=0.99, device=device)
        self.image_distortion = ImageDistortion(perlin_path, min=0, max=2, p=2, device=device, limit=perlin_limit)
        self.format_converter = ImageFormatConverter(device=device)
    
    @property
    def x(self):
        return self._x
        
    def mosaic(self, index):
        *sh, h, w = index.shape
        index = index.view(-1, w)
        mosaic = self.mosaic_builder(self.x[index].float()).clip(0, 255).to(torch.uint8)
        n, c, h, w = mosaic.shape
        return mosaic.view(*sh, c, h, w)
    
    def noisy_mosaic(self, index, generator=None):
        images = self.mosaic(index)
        *sh, c, h, w = images.shape
        images = images.view(-1, c, h, w)
        n, c, h, w = images.shape

        images = self.format_converter.torch_rgb_to_hls(images.float()/255)

        #updating hue
        images[:,0] += torch_uniform(-1/16, 1/16, (n,), device=self.device, generator=generator).view(n, 1, 1)
        images[:,0] %= 1

        #updating lightness
        a = torch_uniform(-0.2, 0.2, (n,), device=self.device, generator=generator).view(n, 1, 1)
        images[:,1] = (1-a.abs())*images[:,1] + torch.maximum(torch.zeros_like(a), a)

        images = self.format_converter.torch_hls_to_rgb(images)

        #spacial distortion
        images = self.image_distortion(images, generator=generator)

        #post processing
        images = (images.clip(0,1)*255).to(torch.uint8)
        
        return images.view(*sh, c, h, w)
