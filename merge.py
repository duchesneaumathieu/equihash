import numpy as np
import os, sys, h5py, argparse
from equihash.utils import timestamp
from equihash.utils.config import load_config
from build_index import build_inverted_index

def merge_hdf5(task, database_name, model, variant, variant_id, load_checkpoint, which, nb_jobs, force, build_index, verbose=False):
    config = load_config(task, model, variant=variant, variant_id=variant_id)
    name = config['name']
    
    checkpoint = 'current' if load_checkpoint is None else f'checkpoint{load_checkpoint}'
    checkpoint_folder = os.path.join('data', 'experiments', task, model, name, checkpoint)
    fingerprints_folder = os.path.join(checkpoint_folder, f'{which}_{database_name}')
    fingerprints_path = os.path.join(fingerprints_folder, f'database_fingerprints.hdf5')
    fingerprints_paths = os.path.join(fingerprints_folder, f'database_fingerprints_{{job_id}}_{nb_jobs}.hdf5')
    
    if os.path.exists(fingerprints_path) and not force:
        raise FileExistsError(fingerprints_path)
    
    if verbose: print(timestamp('Counting number of codes.'), flush=True)
    size = 0
    nbytes = None
    for job_id in range(args.nb_jobs):
        path = fingerprints_paths.format(job_id=job_id)
        with h5py.File(path, 'r') as job_f:
            job_size, job_nbytes = job_f['codes'].shape
            if nbytes is None: nbytes = job_nbytes
            if nbytes != job_nbytes:
                raise RuntimeError(f'nbytes of job {job_id} ({job_nbytes}) differs from the previous jobs ({nbytes})')
            if job_f['size'][0] != job_size:
                raise RuntimeError(f'job {job_id} is uncompleted')
            size += job_size
    if verbose: print(timestamp(f'{size:,} codes found.'), flush=True)

    beg = 0
    codes_shape = (size, nbytes)
    with h5py.File(fingerprints_path, 'w') as f:
        if verbose: print(timestamp(f'Creating dataset: shape={codes_shape}'), flush=True)
        codes = f.create_dataset('codes', shape=codes_shape, dtype=np.uint8)
        for job_id in range(args.nb_jobs):
            path = fingerprints_paths.format(job_id=job_id)
            if verbose: print(timestamp(f'Copying codes from "{path}"... '), end='', flush=True)
            with h5py.File(path, 'r') as job_f:
                job_codes = job_f['codes']
                N = len(job_codes)
                codes[beg: beg+N] = job_codes[:]
                beg += N
            if verbose: print('Done.', flush=True)
    if verbose: print(timestamp('Merge completed.'), flush=True)
        
    if build_index:
        if verbose: print() #newline
        build_inverted_index(task, database_name, model, variant, variant_id, load_checkpoint, which, force=True, verbose=verbose)

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
    parser.add_argument('-n', '--nb_jobs', type=int, required=True,
                        help='The number of jobs used to encode.')
    parser.add_argument('-f', '--force', action='store_true', default=False,
                        help='If this variable is set and an the merged fingerprints exists, it will be overwritten.')
    parser.add_argument('-b', '--build_index', action='store_true', default=False,
                        help='If this variable is set, an inverted index of the database will be build. Incompatible with nb_jobs.')
    
    args = parser.parse_args()
    merge_hdf5(**vars(args), verbose=True)