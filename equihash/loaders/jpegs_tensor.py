import torch
import numpy as np
from PIL import Image

def numpy_like_fancy_indexing(indexable, index, recursion_fn=None):
    if isinstance(index, (int, np.integer)):
        return indexable[index]
    #else assumes index is an iterable
    out = [numpy_like_fancy_indexing(indexable, i) for i in index]
    if recursion_fn is not None:
        out = recursion_fn(out)
    return out

class JPEGLoader:
    def __init__(self, paths, process=None):
        self.paths = paths
        self.process = process
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        x = Image.open(self.paths[index])
        if self.process is not None:
            x = self.process(x)
        return x

class JPEGsTensor:
    def __init__(self, paths, device='cpu', dtype=torch.uint8):
        self.loader = JPEGLoader(paths, process=np.array)
        self.device = device
        self.dtype = dtype
        
    def __len__(self):
        return len(self.loader)
        
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.cpu().numpy()
        array = numpy_like_fancy_indexing(self.loader, index, recursion_fn=np.stack)
        return torch.tensor(array, device=self.device, dtype=self.dtype)
