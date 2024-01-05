import torch

class StraightThroughSign(torch.autograd.Function):
    #return 1 or -1
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
straight_through_sign = StraightThroughSign.apply

def heaviside(x):
    #1 if 0<=x else 0
    return (0<=x).type(x.dtype)

class StraightThroughHeaviside(torch.autograd.Function):
    #return 1 or 0
    @staticmethod
    def forward(ctx, input):
        return heaviside(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
straight_through_heaviside = StraightThroughHeaviside.apply