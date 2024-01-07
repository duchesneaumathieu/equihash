import numpy as np
import os, sys, h5py, argparse

from equihash.utils import timestamp
from equihash.search import InvertedIndex
from equihash.utils.config import load_config

def build_inverted_index(task, database_name, model, variant, variant_id, load_checkpoint, which, force, verbose=False):
    config = load_config(task, model, variant=variant, variant_id=variant_id)
    
    #unpacking useful config item
    name = config['name']
    
    #state path
    load_checkpoint = 'current' if load_checkpoint is None else f'checkpoint{load_checkpoint}'
    checkpoint_folder = os.path.join('data', 'experiments', task, model, name, load_checkpoint)
    
    #fingerprints path
    database_folder = os.path.join(checkpoint_folder, f'{which}_{database_name}')
    fingerprints_path = os.path.join(database_folder, f'database_fingerprints.hdf5')
    inverted_index_path = os.path.join(database_folder, f'database_inverted_index.hdf5')
    
    if os.path.exists(inverted_index_path) and not force:
        raise FileExistsError(inverted_index_path)
    
    with h5py.File(fingerprints_path, 'r') as f:
        codes = f['codes'][:]
    
    N = len(codes)
    nb_heads = 2**(int(np.ceil(np.log2(N)))+1)
    if verbose: print(timestamp(f'Start building the inverted index (nb_heads={nb_heads:,}).'), flush=True)
    ii = InvertedIndex(inverted_index_path, 'w')
    ii.build(nb_heads=nb_heads, codes=codes, algo='linear')
    if verbose: print(timestamp(f'Build completed.'), flush=True)
    ii.save()
    if verbose: print(timestamp(f'Inverted index saved.'), flush=True)
    return ii

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='The experiment\'s task.')
    parser.add_argument('database_name', type=str, help='The database name.')
    parser.add_argument('model', type=str, help='The experiment\'s model.')
    parser.add_argument('-v', '--variant', type=str, required=False, help='The model\'s variant.')
    parser.add_argument('-i', '--variant_id', type=int, required=False, help='The model\'s variant id.')
    parser.add_argument('-l', '--load_checkpoint', type=int, required=False, default=None,
                        help='the network checkpoints (if None is provided, the current state is used).')
    parser.add_argument('-w', '--which', type=str, required=False, default='test',
                        help='Which dataset (train, valid, or test) (default: test).')
    parser.add_argument('-f', '--force', action='store_true', default=False,
                        help='If this variable is set and an inverted index exists, it will be overwritten.')
    
    args = parser.parse_args()
    build_inverted_index(**vars(args), verbose=True)
