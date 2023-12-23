import numpy as np
from .unique_randint import unique_randint

def uint64_to_baseN(N, k, uint64, dtype=np.uint64):
    base = np.zeros(uint64.shape + (k,), dtype=dtype)
    for i in range(k):
        base[..., k-1-i] = uint64//(N**i) % N
    return base

class LabelsGenerator:
    def __init__(self, nb_labels, replacement=True, dtype=np.uint64, rng=np.random):
        self.nb_labels = nb_labels
        self.replacement = replacement
        self.dtype = dtype
        self.rng = rng
        
    def get_rng(self, rng=None):
        return self.rng if rng is None else rng
    
    def get_dtype(self, dtype=None):
        return self.dtype if dtype is None else dtype 
        
    def is_replacement(self, replacement=None):
        return self.replacement if replacement is None else replacement
        
    def _parse_input(self, replacement, dtype, rng):
        return self.is_replacement(replacement), self.get_dtype(dtype), self.get_rng(rng)
    
    def generate_labels(self, nb_samples, replacement=None, dtype=None, rng=None):
        replacement, dtype, rng = self._parse_input(replacement, dtype, rng)
        if replacement: return rng.randint(0, self.nb_labels, nb_samples, dtype=self.dtype)
        else: return unique_randint(0, self.nb_labels, n=1, k=nb_samples, rng=rng)[0]
        
    def generate_positive_pairs(self, nb_pairs, replacement=None, dtype=None, rng=None):
        replacement, dtype, rng = self._parse_input(replacement, dtype, rng)
        if replacement: labels = rng.randint(0, self.nb_labels, nb_pairs, dtype=dtype)
        else: labels = unique_randint(0, self.nb_labels, n=1, k=nb_pairs, dtype=dtype, rng=rng)[0]
        return np.stack([labels, labels], axis=1)
        
    def generate_negative_pairs(self, nb_pairs, replacement=None, dtype=None, rng=None):
        replacement, dtype, rng = self._parse_input(replacement, dtype, rng)
        if replacement: return unique_randint(0, self.nb_labels, n=nb_pairs, k=2, rng=rng)
        else: return unique_randint(0, self.nb_labels, n=1, k=2*nb_pairs, rng=rng).reshape(nb_pairs, 2)
        
    def generate_mixed_pairs(self, nb_positive_pairs, nb_negative_pairs, replacement=None, dtype=None, rng=None):
        nb_pairs = nb_positive_pairs + nb_negative_pairs
        labels = self.generate_negative_pairs(nb_pairs, replacement=replacement, dtype=dtype, rng=rng)
        labels[:nb_positive_pairs, 1] = labels[:nb_positive_pairs, 0]
        return labels
    
    def generate_hinge_triplet(self, nb_triplets, replacement=None, dtype=None, rng=None):
        labels = labels = self.generate_negative_pairs(nb_triplets, replacement=replacement, dtype=dtype, rng=rng)
        return np.concatenate([labels[:,[0]], labels], axis=1)

class MosaicLabelsGenerator:
    def __init__(self, bank_size, mosaic_shape=(), replacement=True, dtype=np.uint64, rng=np.random):
        self.dtype = dtype
        self.bank_size = bank_size
        self.mosaic_shape = mosaic_shape
        self.mosaic_size = np.prod(mosaic_shape, dtype=np.int64)
        if 64 < self.mosaic_size*np.log2(bank_size):
            raise OverflowError('2**64 < nb_labels')
        self.nb_labels = bank_size**self.mosaic_size
        self.labels_generator = LabelsGenerator(self.nb_labels, replacement=replacement, dtype=dtype, rng=rng)
        
    def mosaic_labels_from_labels(self, integer_labels, dtype=None):
        dtype = self.dtype if dtype is None else dtype
        return uint64_to_baseN(
            self.bank_size,
            self.mosaic_size,
            integer_labels,
            dtype=dtype
        ).reshape(*integer_labels.shape, *self.mosaic_shape)
        
    def generate_labels(self, nb_samples, replacement=None, dtype=None, rng=None):
        integer_labels = self.labels_generator.generate_labels(
            nb_samples, replacement=replacement, dtype=dtype, rng=rng)
        return self.mosaic_labels_from_labels(integer_labels, dtype=dtype)
        
    def generate_positive_pairs(self, nb_pairs, replacement=None, dtype=None, rng=None):
        integer_labels = self.labels_generator.generate_positive_pairs(
            nb_pairs, replacement=replacement, dtype=dtype, rng=rng)
        return self.mosaic_labels_from_labels(integer_labels, dtype=dtype)
        
    def generate_negative_pairs(self, nb_pairs, replacement=None, dtype=None, rng=None):
        integer_labels = self.labels_generator.generate_negative_pairs(
            nb_pairs, replacement=replacement, dtype=dtype, rng=rng)
        return self.mosaic_labels_from_labels(integer_labels, dtype=dtype)
        
    def generate_mixed_pairs(self, nb_positive_pairs, nb_negative_pairs, replacement=None, dtype=None, rng=None):
        integer_labels = self.labels_generator.generate_mixed_pairs(
            nb_positive_pairs, nb_negative_pairs, replacement=replacement, dtype=dtype, rng=rng)
        return self.mosaic_labels_from_labels(integer_labels, dtype=dtype)
    
    def generate_hinge_triplet(self, nb_triplets, replacement=None, dtype=None, rng=None):
        integer_labels = self.labels_generator.generate_hinge_triplet(
            nb_triplets, replacement=replacement, dtype=dtype, rng=rng)
        return self.mosaic_labels_from_labels(integer_labels, dtype=dtype)