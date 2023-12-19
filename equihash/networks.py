import torch
import torch.nn as nn
import torch.nn.functional as F
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

class Uint8ResNet(ResNet):
    def forward(self, x):
        #rescale uint8 to [-1, 1]
        x = (x.float()-127.5)/127.5
        return super().forward(x)

class MnistResNet(ResNet):
    def __init__(self, k=1000, version=18, batch_norm=True):
        super().__init__(k=k, version=version, batch_norm=batch_norm)
        #conv1: nb_channel changes from 3 to 1 (RGB to grayscale) and stride changes from 2 to 1
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
        
    def forward(self, x):
        #adding a dummy channel dimension
        return super().forward(x[...,None,:,:])

class MnistMosaicCNN(nn.Module):
    def __init__(self, k=10, batch_norm=True):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 24, 5, 1, 0)
        self.conv2 = nn.Conv2d(24, 32, 3, 1, 0)
        self.conv3 = nn.Conv2d(32, 48, 3, 1, 1)
        self.conv2_bn = nn.BatchNorm2d(24)
        self.conv3_bn = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(1728, 512)
        self.fc2 = nn.Linear(512, k)
        self.bn = None if not batch_norm else nn.BatchNorm1d(k)
    
    def forward(self, x, batch_norm=True):
        #accepting arbitrary shapes
        *shape, h, w = x.shape
        x = x.view(-1, 1, h, w)
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        
        x = self.conv2_bn(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        
        x = self.conv3_bn(x)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        if self.bn is not None and batch_norm:
            x = self.bn(x)
        return x.view(*shape, -1)