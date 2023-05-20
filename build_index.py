import numpy as np
import os, sys, h5py, argparse

script_dir = os.path.dirname(os.path.realpath(__file__))
#sys.path.append(script_dir) #to access the equihash module
from equihash.utils import timestamp
from equihash.search import InvertedIndex

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help='The path of the model\'s configuration json file.')
parser.add_argument('-l', '--load_checkpoint', type=int, required=True, help='The checkpoint\'s step to load.')
parser.add_argument('-w', '--which_set', type=str, required=True, help='"train", "valid", or "test".')
parser.add_argument('-D', '--deterministic', type=int, required=True,
                    help='The procedural database seed.')
parser.add_argument('-m', '--model_name', type=str, required=False, default=None,
                    help='The model name, the default is the config name (without the extension).')
parser.add_argument('-L', '--limit', type=int, default=None, help='The maximal number of mosaic to encode, default is None.')
parser.add_argument('-I', '--indexes_dir', type=str, required=False, default=f'{script_dir}/indexes/',
                    help='The directory to save the index, the default is "{script_dir}/indexes/".')
parser.add_argument('-F', '--fingerprints_dir', type=str, required=False, default=f'{script_dir}/fingerprints/',
                    help='The directory to load the fingerprints, the default is "{script_dir}/fingerprints/".')
args = parser.parse_args()

model_name, extension = os.path.splitext(os.path.basename(args.config))
model_name = model_name if args.model_name is None else args.model_name
checkpoint_name = f'{model_name}_step_{args.load_checkpoint}'
limit_str = '' if args.limit is None else f'{args.limit}'
fingerprints_name = f'{checkpoint_name}_{args.which_set}_{limit_str}fingerprints_{args.deterministic}'
fingerprints_path = os.path.join(f'{args.fingerprints_dir}', f'{fingerprints_name}.hdf5')
index_name = f'{checkpoint_name}_{args.which_set}_{limit_str}index_{args.deterministic}'
index_path = os.path.join(f'{args.indexes_dir}', f'{index_name}.hdf5')

if os.path.exists(index_path):
    raise FileExistsError(index_path)

with h5py.File(fingerprints_path, 'r') as f:
    codes = f['codes'][:args.limit]

N = len(codes)
nb_heads = 2**(int(np.ceil(np.log2(N)))+1)
print(timestamp(f'Start building the inverted index (nb_heads={nb_heads:,}).'), flush=True)
ii = InvertedIndex(index_path, 'w')
ii.build(nb_heads=nb_heads, codes=codes, algo='linear')
print(timestamp(f'Build completed.'), flush=True)
ii.save()
print(timestamp(f'Inverted index saved.'), flush=True)