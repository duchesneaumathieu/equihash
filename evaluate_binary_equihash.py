import numpy as np
import os, h5py, pickle, argparse

from equihash.evaluation import get_equihash_results, get_binary_equihash_results
from equihash.utils.config import load_config
from equihash.utils import timestamp#, avg_integer_bootstrap_format
from equihash.search import InvertedIndex
from encode import encode as enc

def evaluate_binary_equihash(task, database_name, model, variant, variant_id, load_checkpoint, which, device, encode, verbose=False):
    config = load_config(task, model, variant=variant, variant_id=variant_id)
    
    if encode:
        enc(task, database_name, model, variant, variant_id, load_checkpoint, which, device,
           flush_freq=100_000, job_id=0, nb_jobs=None, build_index=True, verbose=verbose)
        if verbose: print() #newline
    
    #unpacking useful config item
    name = config['name']
    seed = config['seed']
    
    #state path
    checkpoint = 'current' if load_checkpoint is None else f'checkpoint{load_checkpoint}'
    checkpoint_folder = os.path.join('data', 'experiments', task, model, name, checkpoint)
    
    #fingerprints path
    database_folder = os.path.join(checkpoint_folder, f'{which}_{database_name}')
    positive_queries_fingerprints_path = os.path.join(database_folder, 'positive_queries_fingerprints.hdf5')
    negative_queries_fingerprints_path = os.path.join(database_folder, 'negative_queries_fingerprints.hdf5')
    hinge_triplets_fingerprints_path = os.path.join(database_folder, 'hinge_triplets_fingerprints.hdf5')
    inverted_index_path = os.path.join(database_folder, f'database_inverted_index.hdf5')
    
    labels_ii_path = os.path.join('data', 'labels', task, f'{database_name}_inverted_index.hdf5')
    ground_truth_path = os.path.join('data', 'labels', task, f'{database_name}_ground_truth.pkl')
    
    if verbose: print(timestamp(f'Working directory: {database_folder}'), flush=True)
    if verbose: print(timestamp(f'Loading inverted index...'), end='', flush=True)
    ii = InvertedIndex(inverted_index_path, 'r').load()
    if verbose: print(f' done.', flush=True)
    
    if verbose: print(timestamp(f'Loading labels inverted index...'), end='', flush=True)
    labels_ii = InvertedIndex(labels_ii_path, 'r').load()
    if verbose: print(f' done.', flush=True)
    
    if verbose: print(timestamp(f'Loading ground truths for positive queries...'), end='', flush=True)
    with open(ground_truth_path, 'rb') as f:
        ground_truths = pickle.load(f)
    if verbose: print(f' done.', flush=True)
    
    if verbose: print(timestamp(f'Loading positive queries...'), end='', flush=True)
    with h5py.File(positive_queries_fingerprints_path, 'r') as f:
        positive_queries = f['codes'][:]
    if verbose: print(f' done.', flush=True)
    
    if verbose: print(timestamp(f'Loading negative queries...'), end='', flush=True)
    with h5py.File(negative_queries_fingerprints_path, 'r') as f:
        negative_queries = f['codes'][:]
    if verbose: print(f' done.', flush=True)
        
    if verbose: print(timestamp(f'Loading Hinge triplets...'), end='', flush=True)
    with h5py.File(hinge_triplets_fingerprints_path, 'r') as f:
         hinge_triplets = f['codes'][:]
    if verbose: print(f' done.', flush=True)
    
    if verbose: print() #newline
    equihash_results = get_equihash_results(ii, labels_ii, positive_queries, negative_queries, ground_truths, verbose=verbose)
    if verbose: print(timestamp(f'Computing binary equihash results...'), end='', flush=True)
    binary_equihash_results = get_binary_equihash_results(hinge_triplets, device=device)
    if verbose: print(f' done.', flush=True)
        
    results = (equihash_results, binary_equihash_results)
    results_path = os.path.join(database_folder, f'results.pkl')
    if verbose: print(timestamp(f'Saving results {results_path}'), flush=True)
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
        
    if verbose:
        print() #newline
        equihash_results.print_results()
        print('\nBinary equihash results:')
        binary_equihash_results.print_results(prefix='\t')

if __name__ == '__main__':
    #TODO confidence intervals
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
    parser.add_argument('-d', '--device', type=str, default='cuda', help='The device to use (cpu or cuda) (default: cuda).')
    parser.add_argument('-e', '--encode', action='store_true', default=False,
                        help='If this variable is set, the fingerprints will be encoded first.')
    
    args = parser.parse_args()
    evaluate_binary_equihash(**vars(args), verbose=True)