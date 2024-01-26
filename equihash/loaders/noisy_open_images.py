import numpy as np
import pickle, torch, h5py, os
from .jpegs_tensor import JPEGsTensor
from .abstractloader import AbstractLoader, AbstractMosaicLoader

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
    def __init__(self, perlin_path, device='cpu', limit=None):
        with h5py.File(perlin_path, 'r') as f:
            self.perlin1 = torch.tensor(f['resolution1'][:limit], device=device)
            self.perlin2 = torch.tensor(f['resolution2'][:limit], device=device)
        h, w = self.shape = self.perlin1.shape[1:]
        self.grid = build_grid(*self.shape, device=device)
        
        max_perlin = 127/320 + 127/640
        self.h_mask = ((1 - torch.linspace(-1, 1, h, device=device).abs()) / max_perlin).clip(0, 1)
        self.w_mask = ((1 - torch.linspace(-1, 1, w, device=device).abs()) / max_perlin).clip(0, 1)
        self.grid = build_grid(*self.shape, device=device)

    def spacial_distortion(self, images, flow_field):
        flow_field = flow_field.clone()
        flow_field[..., 1] *= self.h_mask[None, :, None]
        flow_field[..., 0] *= self.w_mask[None, None, :]
        distorted_grid = self.grid + flow_field
        return torch.nn.functional.grid_sample(images, distorted_grid, align_corners=False, padding_mode='border')
    
    def sample_perlin_noise(self, n, generator):
        n11 = self.perlin1[torch.randint(len(self.perlin1), (n,), device=generator.device, generator=generator)]
        n22 = self.perlin2[torch.randint(len(self.perlin2), (n,), device=generator.device, generator=generator)]
        return n11/320 + n22/640
    
    def __call__(self, images, generator):
        n, _, h, w = images.shape
        flow_field = self.sample_perlin_noise(2*n, generator).view(n,2,h,w).permute(0,2,3,1)
        return self.spacial_distortion(images, flow_field)

class NoisyOpenImages(AbstractLoader):
    def __init__(self, open_images_path, perlin_noise_path,
                 size=None, which='train', device='cpu', HWC=False, dtype=torch.float32, seed=None,
                 images_limit=None, perlin_limit=None, load=True):
        super().__init__(size=size, which=which, dtype=dtype, device=device, seed=seed)
        if (not load) and (not HWC):
            raise NotImplementedError('load=False and HWC=False not implemented.')
        
        self.HWC = HWC
        hdf5_path = os.path.join(open_images_path, 'hdf5', f'{which}.hdf5')
        if load and os.path.exists(hdf5_path):
            with h5py.File(hdf5_path, 'r') as f:
                imgs = f['images'][:images_limit]
            self._x = torch.tensor(imgs, device=device, dtype=torch.uint8)
            if not HWC: self._x = self._x.permute(0, 3, 1, 2) #HWC --> CHW
            self._loaded = True
        else:
            split_path = os.path.join(open_images_path, 'splits', f'{which}_split.pkl')
            with open(split_path, 'rb') as f:
                files = pickle.load(f)[:images_limit]
            imgs_path = [os.path.join(open_images_path, 'OpenImages180x180', file) for file in files]
            self._x = JPEGsTensor(imgs_path, device=device, dtype=torch.uint8)
            self._loaded = False
            
        self.image_distortion = ImageDistortion(perlin_noise_path, device=device, limit=perlin_limit)
        self.format_converter = ImageFormatConverter(device=device)
    
    @property
    def loaded(self):
        return self._loaded
    
    @property
    def x(self):
        return self._x
        
    def unnoised_batch_from_labels(self, index):
        sh = index.shape
        x = self.x[index.flatten()]
        if self.HWC: x = x.permute(0, 3, 1, 2) #HWC --> CHW
        return x.view(*sh, *x.shape[1:])
    
    def batch_from_labels(self, index, nb_instances=None, torch_generator=None, numpy_generator=None):
        torch_generator = self.torch_generator if torch_generator is None else torch_generator
        numpy_generator = self.numpy_generator if numpy_generator is None else numpy_generator
        k = 1 if nb_instances is None else nb_instances
        index = torch.stack(k*[index], dim=1)
        
        images = self.unnoised_batch_from_labels(index)
        *sh, c, h, w = images.shape
        images = images.view(-1, c, h, w)
        n, c, h, w = images.shape
        images = images.float()/255
        
        images = self.format_converter.torch_rgb_to_hls(images)
        
        #updating hue
        images[:,0] += torch_uniform(-1/16, 1/16, (n,), device=self.device, generator=torch_generator).view(n, 1, 1)
        images[:,0] %= 1
        
        #updating lightness
        a = torch_uniform(-0.2, 0.2, (n,), device=self.device, generator=torch_generator).view(n, 1, 1)
        images[:,1] = (1-a.abs())*images[:,1] + torch.maximum(torch.zeros_like(a), a)
        
        #updating saturation
        #a = torch_uniform(-0.1, 0.1, (n,), device=self.device, generator=torch_generator).view(n, 1, 1)
        #images[:,2] = (1-a.abs())*images[:,2] + torch.maximum(torch.zeros_like(a), a)
        
        images = self.format_converter.torch_hls_to_rgb(images)
        
        #spacial distortion
        images = self.image_distortion(images, generator=torch_generator)
        
        #post processing
        images = (images.clip(0,1)*255).to(torch.uint8)
        if self.HWC:
            images = images.permute(0, 2, 3, 1)
            images = images.view(*sh, h, w, c)
        else:
            images = images.view(*sh, c, h, w)
        
        if nb_instances is None:
            images = images[:,0]
        
        return images

    
class NoisyOpenImagesMosaic(AbstractMosaicLoader):
    def __init__(self, open_images_path, perlin_noise_path,
                 size=None, which='train', device='cpu', HWC=False, dtype=torch.float32, seed=None,
                 images_limit=None, perlin_limit=None, load=True):
        super().__init__(height=1, width=2, size=size, which=which, dtype=dtype, device=device, seed=seed)
        if (not load) and (not HWC):
            raise NotImplementedError('load=False and HWC=False not implemented.')
        
        self.HWC = HWC
        hdf5_path = os.path.join(open_images_path, 'hdf5', f'{which}.hdf5')
        if load and os.path.exists(hdf5_path):
            with h5py.File(hdf5_path, 'r') as f:
                imgs = f['images'][:images_limit]
            self._x = torch.tensor(imgs, device=device, dtype=torch.uint8)
            if not HWC: self._x = self._x.permute(0, 3, 1, 2) #HWC --> CHW
            self._loaded = True
        else:
            split_path = os.path.join(open_images_path, 'splits', f'{which}_split.pkl')
            with open(split_path, 'rb') as f:
                files = pickle.load(f)[:images_limit]
            imgs_path = [os.path.join(open_images_path, 'OpenImages180x180', file) for file in files]
            self._x = JPEGsTensor(imgs_path, device=device, dtype=torch.uint8)
            self._loaded = False
            
        self.mosaic_builder = Overlap2x1Mosaic(image_size=180, mosaic_size=270, p=0.99, device=device)
        self.image_distortion = ImageDistortion(perlin_noise_path, device=device, limit=perlin_limit)
        self.format_converter = ImageFormatConverter(device=device)
    
    @property
    def loaded(self):
        return self._loaded
    
    @property
    def x(self):
        return self._x
        
    def unnoised_batch_from_labels(self, index):
        *sh, h, w = index.shape
        index = index.view(-1, w)
        x = self.x[index]
        if self.HWC: x = x.permute(0, 1, 4, 2, 3) #HWC --> CHW
        mosaic = self.mosaic_builder(x.float()).clip(0, 255).to(torch.uint8)
        n, c, h, w = mosaic.shape
        return mosaic.view(*sh, c, h, w)
    
    def batch_from_labels(self, index, nb_instances=None, torch_generator=None, numpy_generator=None):
        torch_generator = self.torch_generator if torch_generator is None else torch_generator
        numpy_generator = self.numpy_generator if numpy_generator is None else numpy_generator
        k = 1 if nb_instances is None else nb_instances
        index = torch.stack(k*[index], dim=1)
        
        images = self.unnoised_batch_from_labels(index)
        *sh, c, h, w = images.shape
        images = images.view(-1, c, h, w)
        n, c, h, w = images.shape
        images = images.float()/255
        
        images = self.format_converter.torch_rgb_to_hls(images)
        
        #updating hue
        images[:,0] += torch_uniform(-1/16, 1/16, (n,), device=self.device, generator=torch_generator).view(n, 1, 1)
        images[:,0] %= 1
        
        #updating lightness
        a = torch_uniform(-0.2, 0.2, (n,), device=self.device, generator=torch_generator).view(n, 1, 1)
        images[:,1] = (1-a.abs())*images[:,1] + torch.maximum(torch.zeros_like(a), a)
        
        #updating saturation #this kind of saturation creates artefacts
        #a = torch_uniform(-0.1, 0.1, (n,), device=self.device, generator=torch_generator).view(n, 1, 1)
        #images[:,2] = (1-a.abs())*images[:,2] + torch.maximum(torch.zeros_like(a), a)
        
        images = self.format_converter.torch_hls_to_rgb(images)
        
        #spacial distortion
        images = self.image_distortion(images, generator=torch_generator)
        
        #post processing
        images = (images.clip(0,1)*255).to(torch.uint8)
        if self.HWC:
            images = images.permute(0, 2, 3, 1)
            images = images.view(*sh, h, w, c)
        else:
            images = images.view(*sh, c, h, w)
        
        if nb_instances is None:
            images = images[:,0]
        
        return images
