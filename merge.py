import numpy as np
import os, sys, h5py, argparse

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_dir) #to access the equihash module
from equihash.utils import timestamp

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help='The path of the model\'s configuration json file.')
parser.add_argument('-l', '--load_checkpoint', type=int, required=True, help='The checkpoint\'s step to load.')
parser.add_argument('-w', '--which_set', type=str, required=True, help='"train", "valid", or "test".')
parser.add_argument('-D', '--deterministic', type=int, required=True,
                    help='The procedural database seed.')
parser.add_argument('-m', '--model_name', type=str, required=False, default=None,
                    help='The model name, the default is the config name (without the extension).')
parser.add_argument('-F', '--fingerprints_dir', type=str, required=False, default=f'{script_dir}/fingerprints/',
                    help='The directory to save the fingerprints, the default is "{script_dir}/fingerprints/".')
parser.add_argument('-L', '--limit', type=int, default=None, help='The maximal number of mosaic to encode, default is None.')
parser.add_argument('-n', '--nb_jobs', type=int, default=None,
                    help=('The number of jobs used to encode the mosaics. The set of mosaics will be divided'
                    ' into nb_jobs chunks of equal sizes and each job will encode one chunk. The default is one job.'))
args = parser.parse_args()

model_name, extension = os.path.splitext(os.path.basename(args.config))
model_name = model_name if args.model_name is None else args.model_name
checkpoint_name = f'{model_name}_step_{args.load_checkpoint}'
limit_str = '' if args.limit is None else f'{args.limit}'
fingerprints_name = f'{checkpoint_name}_{args.which_set}_{limit_str}fingerprints_{args.deterministic}'
fingerprints_path = os.path.join(f'{args.fingerprints_dir}', f'{fingerprints_name}.hdf5')
fingerprints_paths = os.path.join(f'{args.fingerprints_dir}', f'{fingerprints_name}_{{job_id}}_{args.nb_jobs}.hdf5')

if os.path.exists(fingerprints_path):
    raise FileExistsError(fingerprints_path)

print(timestamp('Counting number of codes.'), flush=True)
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
print(timestamp(f'{size:,} codes found.'), flush=True)

beg = 0
codes_shape = (size, nbytes)
with h5py.File(fingerprints_path, 'w') as f:
    print(timestamp(f'Creating dataset: shape={codes_shape}'), flush=True)
    codes = f.create_dataset('codes', shape=codes_shape, dtype=np.uint8)
    for job_id in range(args.nb_jobs):
        path = fingerprints_paths.format(job_id=job_id)
        print(timestamp(f'Copying codes from "{path}"... '), end='', flush=True)
        with h5py.File(path, 'r') as job_f:
            job_codes = job_f['codes']
            N = len(job_codes)
            codes[beg: beg+N] = job_codes[:]
            beg += N
        print('Done.', flush=True)
print('Merge completed.', flush=True)