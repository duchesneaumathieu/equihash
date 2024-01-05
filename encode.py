import numpy as np
import os, sys, json, h5py, torch, argparse

from equihash.utils import timestamp
from equihash.utils.config import load_config, build_loader, build_network
from equihash.utils.fingerprints import Fingerprints
from equihash.databases import ProceduralLabelsDB

def _encode(net, loader, database_labels_path, which_labels, fingerprints_path, seed, flush_freq, job_id, nb_jobs):
    #which_labels in ('database', 'positive_queries', 'negative_queries')
    if nb_jobs is None:
        nb_jobs = 1
    if not (0 <= job_id < nb_jobs):
        raise ValueError(f'job_id out of range for nb_jobs={nb_jobs}')
    
    nbits = net.k
    if nbits % 8 != 0:
        raise NotImplementedError('nbits must be a multiple of 8')
    nbytes = nbits//8
    
    with h5py.File(database_labels_path, 'r') as f:
        n = len(f[which_labels])
        if n % nb_jobs != 0:
            raise NotImplementedError(f'nb_jobs ({nb_jobs}) must divide the number of mosaics ({n})')
        job_size = n // nb_jobs
        labels = f[which_labels][:]
    
    database = ProceduralLabelsDB(labels, loader, seed=seed)
    if flush_freq % database.chunk_size != 0:
        raise ValueError(f'flush_freq ({flush_freq}) must be a multiple of chunk_size ({database.chunk_size})')
    if job_size % database.chunk_size != 0:
        raise NotImplementedError(f'job_size ({job_size}) must be a multiple of chunk_size ({database.chunk_size})')
    
    job_beg, job_end = job_id*job_size, (job_id+1)*job_size
    print(timestamp(f'Job {job_id+1}/{nb_jobs} (job_id={job_id}): encoding from {job_beg:,} to {job_end:,}'), flush=True)
    
    if not os.path.exists(fingerprints_path):
        with h5py.File(fingerprints_path, 'x') as f:
            f.create_dataset('size', shape=(1,), dtype=np.uint32)
            f.create_dataset('codes', shape=(job_size, nbytes), dtype=np.uint8)

    bs = database.chunk_size
    fingerprints = Fingerprints(device=args.device)
    which_codes = 'codes' if which_labels=='labels' else which_labels
    with torch.no_grad(), h5py.File(fingerprints_path, 'r+') as f:
        ncodes = f['size'][0]
        codes = f['codes']
        print(timestamp(f'Starts encoding {which_labels} from {job_beg+ncodes:,} to {job_end:,}'), flush=True)
        while ncodes < job_size:
            batch = database[job_beg+ncodes:job_beg+ncodes+bs]
            codes[ncodes:ncodes+bs] = fingerprints.bool_to_uint8(0<net(batch)).cpu().numpy()
            ncodes += bs
            if ncodes%flush_freq==0:
                codes.flush() #flush codes before we set size
                f['size'][0] = ncodes; f.flush() #saving progress
                print(timestamp(f'{ncodes:,} documents encoded'), flush=True)

def encode(task, database_name, model, variant, variant_id, load_checkpoint, which, device, flush_freq, job_id, nb_jobs):
    config = load_config(task, model, variant=variant, variant_id=variant_id)

    #unpacking useful config item
    name = config['name']
    seed = config['seed']
    
    #making sure everything is deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    
    #state path
    load_checkpoint = 'current' if load_checkpoint is None else f'checkpoint{load_checkpoint}'
    checkpoint_folder = os.path.join('data', 'experiments', task, model, name, load_checkpoint)
    
    #fingerprints path
    fingerprints_folder = os.path.join(checkpoint_folder, f'{which}_{database_name}')
    job_suffix = '' if nb_jobs is None else f'_{job_id}_{nb_jobs}'
    fingerprints_path = os.path.join(fingerprints_folder, f'database_fingerprints{job_suffix}.hdf5')
    os.makedirs(fingerprints_folder, exist_ok=True)
    
    #labels path 
    database_labels_path = os.path.join('data', 'labels', task, f'{database_name}.hdf5')

    print(timestamp(f'Building loader [{which}]...'), end='', flush=True)
    loader = build_loader(config, which=which, device=device, seed=seed)
    print(f' done.', flush=True)
    
    state_path = os.path.join(checkpoint_folder, 'state.pth')
    print(timestamp(f'Loading model {task}:{model}:{name}:{load_checkpoint}...'), end='', flush=True)
    net = build_network(config, device=device)
    state = torch.load(state_path)
    net.load_state_dict(state['net'])
    net.eval()
    print(f' done.', flush=True)
    
    print() #newline
    _encode(net, loader, database_labels_path, 'database', fingerprints_path, seed, flush_freq, job_id, nb_jobs)
    if nb_jobs is None:
        print() #newline
        positive_path = os.path.join(fingerprints_folder, 'positive_queries_fingerprints.hdf5')
        _encode(net, loader, database_labels_path, 'positive_queries', positive_path, seed, flush_freq, job_id=0, nb_jobs=None)
        print() #newline
        negative_path = os.path.join(fingerprints_folder, 'negative_queries_fingerprints.hdf5')
        _encode(net, loader, database_labels_path, 'negative_queries', negative_path, seed, flush_freq, job_id=0, nb_jobs=None)

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
    parser.add_argument('-d', '--device', type=str, default='cuda', help='The device to use (cpu or cuda) (default: cuda).')
    parser.add_argument('-f', '--flush_freq', type=int, default=100_000,
                        help='The frequency at which to flush the hdf5 (default: 100_000).')
    parser.add_argument('-j', '--job_id', type=int, default=0,
                        help='The job id (from 0 to nb_jobs-1). This process will encode i-th chunk (default: 0).') 
    parser.add_argument('-n', '--nb_jobs', type=int, default=None,
                        help=('The number of jobs used to encode. The set of labels will be divided'
                        ' into nb_jobs chunks of equal sizes and each job will encode one chunk. The default is one job.'))
    
    args = parser.parse_args()
    encode(**vars(args))
