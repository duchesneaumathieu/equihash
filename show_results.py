import os, pickle, argparse

from equihash.utils.config import load_results
from equihash.utils import timestamp

def print_results(task, database_name, model, variant, variant_id, load_checkpoint, which):
    equihash_results, binary_equihash_results = load_results(task, database_name, model, variant, variant_id, load_checkpoint, which)
    
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
    
    args = parser.parse_args()
    print_results(**vars(args))
