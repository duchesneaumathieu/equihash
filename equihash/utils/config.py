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