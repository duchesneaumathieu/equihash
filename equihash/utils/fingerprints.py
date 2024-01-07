import torch

class Fingerprints:
    def __init__(self, device):
        self.device = device
        self.uint8_onehot = torch.tensor([128,  64,  32,  16,   8,   4,   2,   1], dtype=torch.uint8, device=device)
        self.shift = torch.tensor([7, 6, 5, 4, 3, 2, 1, 0], dtype=torch.uint8, device=device)

    def bool_to_uint8(self, bool_codes):
        B, N = bool_codes.shape #8 must divide N
        bool_codes = bool_codes.view(B, N//8, 8)
        return (bool_codes * self.uint8_onehot).sum(dim=2, dtype=torch.uint8)
    
    def uint8_to_bool(self, uint8_codes):
        B, N = uint8_codes.shape
        bool_codes = uint8_codes[...,None].__and__(self.uint8_onehot) >> self.shift
        return bool_codes.view(B, N*8).to(bool)

    def uint8_is_equal(self, x, y):
        return (x==y).all(dim=-1)

    def __popcll(self, x, dtype=torch.int32):
        pcll = x%2
        x = x>>1
        for i in range(7):
            pcll += x%2
            x >>= 1
        del x
        out = torch.zeros((len(pcll),), dtype=dtype, device=pcll.device)
        for i in range(pcll.shape[-1]):
            out += pcll[..., i]
        return out #this is equal to pcll.sum(dim=-1) but with less memory

    def uint8_hamming_distance(self, x, y):
        return self.__popcll(x.__xor__(y))
