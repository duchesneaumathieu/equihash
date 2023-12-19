import os, json, collections.abc
from copy import deepcopy

import equihash.loaders as loaders
import equihash.networks as networks
import equihash.trainers as trainers

def build_network(config, device):
    cls = networks.__dict__[config['network_class']]
    return cls(**config['network_kwargs']).to(device=device)

def build_loader(config, which='train', device='cpu', seed=None):
    cls = loaders.__dict__[config['loader_class']]
    return cls(**config['loader_kwargs'], which=which, device=device, seed=seed)

def build_trainer(config, net, train_loader):
    cls = trainers.__dict__[config['trainer_class']]
    return cls(net, train_loader, **config['trainer_kwargs'])

def deep_update(d, u, preserve_type=False, inplace=False):
    if not inplace:
        d = deepcopy(d)
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_update(
                d.get(k, {}), v,
                preserve_type=preserve_type,
                inplace=True
            )
        elif k in d and preserve_type:
            d[k] = type(d[k])(v)
        else:
            d[k] = v
    return d

def rebuild_flat_dict(keys, values):
    rebuild = dict()
    for key, v in zip(keys, values):
        d = rebuild
        *nodes, leaf = key.split(':')
        for n in nodes:
            if n not in d: d[n] = dict()
            d = d[n]
        d[leaf] = v
    return rebuild

def load_variant(task, model, variant, variant_id):
    variant_path = os.path.join('configs', task, model, f'{variant}.json')
    with open(variant_path, 'r') as f:
        key, *variants = json.load(f)
    if len(variants) <= variant_id:
        raise ValueError(f'{variant_path} contains only {len(variants)} variants. However, variant_id={variant_id}')
    return rebuild_flat_dict(key, variants[variant_id])

def build_args(args):
    keys = list()
    values = list()
    for *key, value in [a.split(':') for a in args]:
        keys.append(':'.join(key))
        values.append(value)
    return rebuild_flat_dict(keys, values)        

def load_config(task, model, variant=None, variant_id=None, args=None):
    taskconfig_path = os.path.join('configs', task, 'taskconfig.json')
    with open(taskconfig_path, 'r') as f:
        taskconfig = json.load(f)

    modelconfig_path = os.path.join('configs', task, model, 'modelconfig.json')
    with open(modelconfig_path, 'r') as f:
        modelconfig = json.load(f)
        
    config = deep_update(taskconfig, modelconfig)
        
    if variant is not None:
        if variant_id is None:
            raise ValueError('variant_id must be set when variant is set.')
        variant = load_variant(task, model, variant, variant_id)
        config = deep_update(config, variant)
        
    if args is not None:
        args = build_args(args)
        config = deep_update(config, args, preserve_type=True)
        
    return config