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