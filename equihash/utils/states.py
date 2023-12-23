import os, torch
from equihash.utils import dict_diff

def load_state(path, config, net, trainer, train_results, valid_results, force=False, verbose=False):
    if path.startswith('/dev/null'):
        return config
    
    state = torch.load(path)
    config_diff = '\n    '.join(f'{k}: {v}' for k, v in dict_diff(state['config'], config).items())
    if config_diff:
        msg = f'Configs difference(s):\n    {config_diff}'
        if not force:
            raise ValueError(msg)
        if verbose:
            print(msg)
        
    net.load_state_dict(state['net'])
    trainer.load_state_dict(state['trainer'])
    train_results.load_state_dict(state['train_results'])
    valid_results.load_state_dict(state['valid_results'])
    #torch.set_rng_state(state['cpu_rng'])
    #torch.cuda.set_rng_state(state['gpu_rng'])
    return state['config']

def save_state(path, config, net, trainer, train_results, valid_results):
    if path.startswith('/dev/null'): return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {'config': config,
             'net': net.state_dict(),
             'trainer': trainer.state_dict(),
             'train_results': train_results.state_dict(),
             'valid_results': valid_results.state_dict(),
             #'cpu_rng': torch.get_rng_state(),
             #'gpu_rng': torch.cuda.get_rng_state()
            }
    torch.save(state, path)