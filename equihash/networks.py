import torch
from torchvision.models.resnet import ResNet as CustomResNet, BasicBlock, Bottleneck

_types = {
    18: (BasicBlock, [2, 2, 2, 2]), #ResNet18
    34: (BasicBlock, [3, 4, 6, 3]), #ResNet34
    50: (Bottleneck, [3, 4, 6, 3]), #ResNet50
}

class ResNet(CustomResNet):
    def __init__(self, k=1000, version=18, batch_norm=True):
        super().__init__(*_types[version], num_classes=k)
        self.k = k
        self.version = version
        self.batch_norm = batch_norm
        if batch_norm:
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(512, k, bias=None),
                torch.nn.BatchNorm1d(k, affine=True),
            )
            
    def forward(self, x):
        #accepting arbitrary shapes
        *shape, c, h, w = x.shape
        out = super().forward(x.view(-1, c, h, w))
        return out.view(*shape, -1)

class MnistResNet(ResNet):
    def __init__(self, k=1000, version=18, batch_norm=True):
        super().__init__(k=k, version=version, batch_norm=batch_norm)
        #conv1: nb_channel changes from 3 to 1 (RGB to grayscale) and stride changes from 2 to 1
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
    
    def forward(self, x):
        #adding a dummy channel dimension
        return super().forward(x[...,None,:,:])

class Uint8ResNet(ResNet):
    def forward(self, x):
        #rescale uint8 to [-1, 1]
        x = (x.float()-127.5)/127.5
        return super().forward(x)