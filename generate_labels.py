import os, h5py, pickle, argparse
import numpy as np
from equihash.utils import timestamp
from equihash.search import InvertedIndex, InvertedIndexBuckets
from equihash.utils.unique_randint import unique_randint

def uint64_to_uint8(uint64):
    uint8 = np.zeros(uint64.shape + (8,), dtype=np.uint8)
    for i in range(8):
        uint8[..., 7-i] = uint64>>(i*8) #no need to % 256 because uint8.dtype.type is np.uint8
    return uint8

def uint8_to_uint64(uint8):
    uint64 = np.zeros(uint8.shape[:-1], dtype=np.uint64)
    for i in range(8):
        uint64 += uint8[...,7-i].astype(np.uint64)<<(i*8)
    return uint64

def uint64_to_baseN(N, k, uint64, dtype=np.uint64):
    base = np.zeros(uint64.shape + (k,), dtype=dtype)
    for i in range(k):
        base[..., k-1-i] = uint64//(N**i) % N
    return base

class LabelsGenerator:
    def __init__(self, bank_size, mosaic_shape):
        self.bank_size = bank_size
        self.mosaic_shape = mosaic_shape
        self.mosaic_size = np.prod(mosaic_shape) if mosaic_shape else 1
        assert 64/np.log2(bank_size) > self.mosaic_size, 'nb_labels >= 2**64'
        self.nb_labels = bank_size**self.mosaic_size
        self.dtype = [np.uint8, np.uint16, np.uint32, np.uint64][np.argmax(np.log2(bank_size) < [8,16,32,64])]
        
    def generate_labels_uint64(self, nb_samples, rng):
        return rng.randint(0, self.nb_labels, nb_samples, dtype=np.uint64)
    
    def generate_labels_uint8(self, nb_samples, rng):
        labels_uint64 = self.generate_labels_uint64(nb_samples, rng)
        return uint64_to_uint8(labels_uint64)
    
    def labels_from_uint64(self, labels_uint64):
        return uint64_to_baseN(
            self.bank_size,
            self.mosaic_size,
            labels_uint64,
            dtype=self.dtype
        ).reshape(*labels_uint64.shape, *self.mosaic_shape)
    
    def labels_from_uint8(self, labels_uint8):
        labels_uint64 = uint8_to_uint64(labels_uint8)
        return self.labels_from_uint64(labels_uint64)

    def generate_labels(self, nb_samples, rng):
        labels_uint64 = self.generate_labels_uint64(nb_samples, rng)
        return self.labels_from_uint64(labels_uint64)
    
def code_generator(labels_generator, batch_size, rng):
    while True:
        batch = labels_generator.generate_labels_uint8(batch_size, rng)
        for code in batch: yield code

def generate_negative_queries(nb, ii, labels_generator, rng, batch_size=1_000):
    codes_size = ii.codes.shape[1]
    negative_queries = np.zeros((nb, codes_size), dtype=np.uint8)
    nb_queries = 0
    generator = code_generator(labels_generator, batch_size=batch_size, rng=rng)
    while nb_queries < nb:
        code = next(generator)
        collision = bool(ii[code])
        if not collision:
            negative_queries[nb_queries] = code
            nb_queries += 1
    return negative_queries

def int_generator(a, b, batch_size, rng):
    while True:
        batch = rng.randint(a, b, batch_size)
        for i in batch: yield i

def generate_positive_queries(nb, ii, rng, batch_size=1_000):
    nb_codes, codes_size = ii.codes.shape
    positive_queries = np.zeros((nb, codes_size), dtype=np.uint8)
    nb_queries = 0
    ground_truth = list()
    generator = int_generator(0, nb_codes, batch_size=batch_size, rng=rng)
    while nb_queries < nb:
        i = next(generator)
        code = ii.codes[i]
        indexes = ii[code]
        collision = bool(indexes)
        if collision and i==min(indexes): #i==min(indexes) to reject duplicates so that every labels has the same probability
            positive_queries[nb_queries] = code
            ground_truth.append(indexes)
            nb_queries += 1
    return positive_queries, ground_truth

def main(task, name, bank_size, mosaic_shape, database_size,
         nb_positive_queries, nb_negative_queries, nb_negative_pairs, seed, force, verbose=False):
    path = os.path.join('data', 'labels', task, f'{name}.hdf5')
    ii_path = os.path.join('data', 'labels', task, f'{name}_inverted_index.hdf5')
    iib_path = os.path.join('data', 'labels', task, f'{name}_inverted_index_buckets.hdf5')
    ground_truth_path = os.path.join('data', 'labels', task, f'{name}_ground_truth.pkl')
    file_exists = os.path.exists(path) or os.path.exists(ii_path) or os.path.exists(ground_truth_path)
    if file_exists and not force:
        p = os.path.join('data', 'labels', task, f'{name}*')
        raise FileExistsError(f'{p} exists, use --force to overwrite.')
    
    rng = np.random.RandomState(seed)
    labels_gen = LabelsGenerator(bank_size=bank_size, mosaic_shape=mosaic_shape)
    if verbose: print(timestamp(f'Labels dtype={np.zeros((), dtype=labels_gen.dtype).dtype}.'), flush=True)
    
    msg = f'Generating the {database_size:,} labels (seed={seed})...'
    if verbose: print(timestamp(msg), end='\r', flush=True)
    codes = np.zeros((database_size, 8), dtype=np.uint8)
    labels = np.zeros((database_size, *mosaic_shape), dtype=labels_gen.dtype)
    batch_size = 1_000_000
    nb_batches = database_size // batch_size + (database_size % batch_size != 0)
    for i in range(nb_batches): #doing it in batch for big databases.
        if verbose: print(timestamp(f'{msg} {int(100*i/nb_batches)}%'), end='\r')
        a, b = i*batch_size, min(database_size, (i+1)*batch_size)
        labels_uint64 = labels_gen.generate_labels_uint64(b-a, rng)
        codes[a:b] = uint64_to_uint8(labels_uint64)
        labels[a:b] = labels_gen.labels_from_uint64(labels_uint64)
    if verbose: print(timestamp(f'{msg} Done.'), flush=True)
        
    if verbose: print(timestamp(f'Saving database at {path}...'), end='', flush=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, 'w') as f:
        f.create_dataset('database', data=labels)
    del labels #free some memory, useful for a large database.
    if verbose: print(' Done.', flush=True)
    
    if verbose: print(timestamp('Building the inverted index...'), end='', flush=True)
    ii = InvertedIndex(path=ii_path, mode='w').build(codes=codes, nb_heads=2*database_size)
    if verbose: print(' Done.', flush=True)
        
    if verbose: print(timestamp(f'Start building the inverted index buckets.'), flush=True)
    iib = InvertedIndexBuckets(iib_path, 'w')
    iib.build(ii)
    if verbose: print(timestamp(f'Build completed.'), flush=True)
    iib.save()
    del iib #free memory
    if verbose: print(timestamp(f'Inverted index buckets saved.'), flush=True)
    
    if verbose: print(timestamp(f'Generating {nb_positive_queries:,} positive queries...'), end='', flush=True)
    positive_queries_uint8, ground_truth = generate_positive_queries(nb_positive_queries, ii, rng=rng)
    positive_queries = labels_gen.labels_from_uint8(positive_queries_uint8)
    if verbose: print(' Done.', flush=True)
    
    if verbose: print(timestamp('Verifying positive queries...'), end='', flush=True)
    for code in positive_queries_uint8:
        if not bool(ii[code]):
            if verbose: print(' Error.', flush=True)
            raise ValueError(f'positive query ({code}) not found in the inverted index')
    if verbose: print(' Done.', flush=True)
    
    if verbose: print(timestamp(f'Generating {nb_negative_queries:,} negative queries...'), end='', flush=True)
    negative_queries_uint8 = generate_negative_queries(nb_negative_queries, ii, labels_gen, rng=rng)
    negative_queries = labels_gen.labels_from_uint8(negative_queries_uint8)
    if verbose: print(' Done.', flush=True)
    
    if verbose: print(timestamp('Verifying negative queries...'), end='', flush=True)
    for code in negative_queries_uint8:
        if bool(ii[code]):
            if verbose: print(' Error.', flush=True)
            raise ValueError(f'negative query ({code}) found in the inverted index')
    if verbose: print(' Done.', flush=True)
    
    if verbose: print(timestamp(f'Generating {nb_negative_pairs:,} negative pairs...'), end='', flush=True)
    negative_pairs_uint64 = unique_randint(0, labels_gen.nb_labels, n=nb_negative_pairs, k=2, dtype=np.int64, rng=rng)
    negative_pairs = labels_gen.labels_from_uint64(negative_pairs_uint64)
    if verbose: print(' Done.', flush=True)
    
    if verbose: print(timestamp(f'Saving queries at {path}...'), end='', flush=True)
    with h5py.File(path, 'r+') as f:
        f.create_dataset('positive_queries', data=positive_queries)
        f.create_dataset('negative_queries', data=negative_queries)
        f.create_dataset('negative_pairs', data=negative_pairs)
    if verbose: print(' Done.', flush=True)
    
    if verbose: print(timestamp(f'Saving {ground_truth_path}...'), end='', flush=True)
    with open(ground_truth_path, 'wb') as f:
        pickle.dump(ground_truth, f)
    if verbose: print(' Done.', flush=True)
    
    if verbose: print(timestamp(f'Saving {ii_path}...'), end='', flush=True)
    ii.save()
    if verbose: print(' Done.', flush=True)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='The task name.')
    parser.add_argument('name', type=str, help='The created file name.')
    parser.add_argument('-b', '--bank_size', type=int, help='The size of the data bank.', required=True)
    parser.add_argument('-d', '--database_size', type=int, help='The amount of labels to generate.', required=True)
    parser.add_argument('-p', '--nb_positive_queries', type=int, help='The number of positive queries to generate.', required=True)
    parser.add_argument('-n', '--nb_negative_queries', type=int, help='The number of negative queries to generate.', required=True)
    parser.add_argument('-t', '--nb_negative_pairs', type=int, help='The number of negative pairs to generate.', required=True)
    parser.add_argument('-m', '--mosaic_shape', type=int, nargs='*', help='If needed, provides the shape of the mosaic.', default=[])
    parser.add_argument('-s', '--seed', type=int, help='The seed for the random number generator.', default=0xcafe)
    parser.add_argument('-f', '--force', action='store_true', default=False, help='If set, it overwrite the file if it exists.')

    args = parser.parse_args()
    main(**vars(args), verbose=True)