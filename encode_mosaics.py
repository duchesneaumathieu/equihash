import numpy as np
import os, sys, json, h5py, torch, argparse

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_dir) #to access the equihash module
from equihash.utils import timestamp
from equihash.utils.config import build_loader, build_network
from equihash.utils.fingerprints import Fingerprints
from equihash.databases import ProceduralMosaicDB

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help='The path of the model\'s configuration json file.')
parser.add_argument('-l', '--load_checkpoint', type=int, required=True, help='The checkpoint\'s step to load.')
parser.add_argument('-i', '--index_path', type=str, required=True, help='The path to load the indexes of the mosaics.')
parser.add_argument('-w', '--which_set', type=str, required=True, help='"train", "valid", or "test".')
parser.add_argument('-D', '--deterministic', type=int, required=True,
                    help='The procedural database seed.')
parser.add_argument('-m', '--model_name', type=str, required=False, default=None,
                    help='The model name, the default is the config name (without the extension).')
parser.add_argument('-s', '--states_dir', type=str, required=False, default=f'{script_dir}/states/',
                    help='The directory to load the state\'s checkpoint, the default is "{script_dir}/states/".')
parser.add_argument('-F', '--fingerprints_dir', type=str, required=False, default=f'{script_dir}/fingerprints/',
                    help='The directory to save the fingerprints, the default is "{script_dir}/fingerprints/".')
parser.add_argument('-c', '--chunk_size', type=int, default=500, help='The chunk size of the procedural database.')
parser.add_argument('-f', '--flush_freq', type=int, default=100_000,
                    help='The frequency at which to flush the hdf5, default is 100_000.')
parser.add_argument('-d', '--device', type=str, default='cuda', help='The device to use (cpu or cuda).')
parser.add_argument('-L', '--limit', type=int, default=None, help='The maximal number of mosaic to encode, default is None.')
parser.add_argument('-j', '--job_id', type=int, default=0, help='The job id (from 0 to nb_jobs-1). This process will encode i-th chunk.') 
parser.add_argument('-n', '--nb_jobs', type=int, default=None,
                    help=('The number of jobs used to encode the mosaics. The set of mosaics will be divided'
                    ' into nb_jobs chunks of equal sizes and each job will encode one chunk. The default is one job.'))
args = parser.parse_args()

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(args.deterministic)

model_name, extension = os.path.splitext(os.path.basename(args.config))
model_name = model_name if args.model_name is None else args.model_name
checkpoint_name = f'{model_name}_step_{args.load_checkpoint}'
checkpoint_path = os.path.join(args.states_dir, f'{checkpoint_name}.pth')
limit_str = '' if args.limit is None else f'{args.limit}'
fingerprints_name = f'{checkpoint_name}_{args.which_set}_{limit_str}fingerprints_{args.deterministic}'
job_suffix = '' if args.nb_jobs is None else f'_{args.job_id}_{args.nb_jobs}'
fingerprints_path = os.path.join(f'{args.fingerprints_dir}', f'{fingerprints_name}{job_suffix}.hdf5')

if args.nb_jobs is None:
    args.nb_jobs = 1
if not (0 <= args.job_id < args.nb_jobs):
    raise ValueError(f'job_id out of range for nb_jobs={args.nb_jobs}')

if args.flush_freq % args.chunk_size != 0:
    raise ValueError(f'flush_freq ({args.flush_freq}) must be a multiple of chunk_size ({args.chunk_size})')
    
with open(args.config, 'r') as f:
    config = json.load(f)
net = build_network(config, device=args.device)
state = torch.load(checkpoint_path)
net.load_state_dict(state['net'])
net.eval()
print(timestamp(f'Checkpoint loaded: "{checkpoint_name}"'), flush=True)

nbits = net.k
if nbits % 8 != 0:
    raise NotImplementedError('nbits must be a multiple of 8')
nbytes = nbits//8
    
with h5py.File(args.index_path, 'r') as f:
    n = len(f['index']) if args.limit is None else args.limit
    
    if n % args.nb_jobs != 0:
        raise NotImplementedError(f'nb_jobs ({args.nb_jobs}) must divide the number of mosaics ({n})')
    job_size = n // args.nb_jobs
    
    if job_size % args.chunk_size != 0:
        raise NotImplementedError(f'job_size ({job_size}) must be a multiple of chunk_size ({args.chunk_size})')
    
    index = f['index'][:n]

loader = build_loader(config, which=args.which_set, device=args.device)
db = ProceduralMosaicDB(index, loader, seed=args.deterministic, chunk_size=args.chunk_size)

job_beg, job_end = args.job_id*job_size, (args.job_id+1)*job_size
print(timestamp(f'Job {args.job_id+1}/{args.nb_jobs} (job_id={args.job_id}): encoding from {job_beg:,} to {job_end:,}'), flush=True)

if not os.path.exists(fingerprints_path):
    with h5py.File(fingerprints_path, 'x') as f:
        f.create_dataset('size', shape=(1,), dtype=np.uint32)
        f.create_dataset('codes', shape=(job_size, nbytes), dtype=np.uint8)

bs = db.chunk_size
fingerprints = Fingerprints(device=args.device)
with torch.no_grad(), h5py.File(fingerprints_path, 'r+') as f:
    ncodes = f['size'][0]
    codes = f['codes']
    print(timestamp(f'Starts encoding from {job_beg+ncodes:,} to {job_end:,}'), flush=True)
    while ncodes < job_size:
        batch = db[job_beg+ncodes:job_beg+ncodes+bs]
        codes[ncodes:ncodes+bs] = fingerprints.bool_to_uint8(0<net(batch)).cpu().numpy()
        ncodes += bs
        if ncodes%args.flush_freq==0:
            codes.flush() #flush codes before we set size
            f['size'][0] = ncodes; f.flush() #saving progress
            print(timestamp(f'{ncodes:,} documents encoded'), flush=True)
