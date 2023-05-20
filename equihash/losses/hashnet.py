import torch

softplus = torch.nn.Softplus()
class HashNetLoss:
    def __init__(self, alpha=None, beta=None):
        self.alpha = alpha
        self.beta = beta
        
    def __call__(self, h, s, alpha=None, beta=None):
        #h.shape = (bs, 2, nbits)
        #s.shape = (bs,)
        
        alpha = self.alpha if alpha is None else alpha
        beta = self.beta if beta is None else beta
        
        dot = (alpha*torch.tanh(beta*h)).prod(dim=1).sum(dim=1)
        return softplus(dot) - s*dot