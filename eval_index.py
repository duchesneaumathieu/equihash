import numpy as np
import h5py, pickle, argparse

from equihash.utils import timestamp, avg_integer_bootstrap_format
from equihash.search import InvertedIndex

parser = argparse.ArgumentParser()
parser.add_argument('-I', '--index_path', type=str, required=True, help='The inverted index\'s to load.')
parser.add_argument('-Q', '--queries_path', type=str, required=True,
                    help='The queries\' fingerprints (.h5py) to test against the inverted index.')
parser.add_argument('-D', '--deterministic', type=int, default=None, help='The bootstrap seed. default=None')
parser.add_argument('-L', '--limit', type=int, default=None, help='The maximal number of queries to test, default is None.')
parser.add_argument('-S', '--save_path', type=str, default=None, help='The path used to save the results of each query.')
parser.add_argument('-p', '--precision', type=int, default=None, help='The maximal precision of the results.')
args = parser.parse_args()

z = 0.05
nb_bootstraps = 10_000
bootstrap_rng = np.random.RandomState(args.deterministic)
bootstrap = lambda data: avg_integer_bootstrap_format(
    data, nb_bootstraps=nb_bootstraps, batch_size=1_000, z=z, max_precision=args.precision, rng=bootstrap_rng)

print(timestamp(f'Loading the inverted index... '), end='', flush=True)
ii = InvertedIndex(args.index_path, 'r').load()
print('Done', flush=True)

with h5py.File(args.queries_path, 'r') as f:
    queries = f['codes'][:args.limit]

success = list()
db_sizes = [10**i for i in range(10) if len(queries) <= 10**i <= len(ii.codes)]
perfect_retrieval = {db_size: list() for db_size in db_sizes}
buckets_size = {db_size: list() for db_size in db_sizes}
progress = 0
print(timestamp(f'Evaluating {len(queries):,} queries'))
print(timestamp(f'{progress}% completed'), end='\r', flush=True)
for i, code in enumerate(queries):
    retrieved_index = ii[code]
    is_found = i in retrieved_index
    success.append(is_found)
    for db_size in db_sizes:
        nb_retrieved_index = len([j for j in retrieved_index if j < db_size])
        perfect_retrieval[db_size].append(is_found and nb_retrieved_index==1)
        buckets_size[db_size].append(nb_retrieved_index)
    new_progress = (100*(i+1)) // len(queries)
    if progress < new_progress:
        progress = new_progress
        print(timestamp(f'{progress}% completed'), end='\r', flush=True)
print(flush=True)
success = np.array(success)
perfect_retrieval = {db_size: np.array(v) for db_size, v in perfect_retrieval.items()}
buckets_size = {db_size: np.array(v) for db_size, v in buckets_size.items()}

if args.save_path is not None:
    print(timestamp(f'Saving results'))
    results = {
        'success': success,
        'perfect_retrieval': perfect_retrieval,
        'buckets_size': buckets_size,
    }
    with open(args.save_path, 'wb') as f:
        pickle.dump(results, f)

avg, err = bootstrap(100*np.array(success))
print(timestamp(f'recall: {avg}%±{err}%'))

for db_size in db_sizes:
    avg, err = bootstrap(100*np.array(perfect_retrieval[db_size]))
    print(timestamp(f'PRR@{db_size:,}: {avg}%±{err}%'))
    
for db_size in db_sizes:
    avg, err = bootstrap(np.array(buckets_size[db_size]))
    print(timestamp(f'bsize@{db_size:,}: {avg}±{err}'))