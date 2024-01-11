import torch
import numpy as np
from equihash.losses.straight_through import straight_through_sign, straight_through_heaviside
from equihash.evaluation import saturation_ratio
from equihash.utils import ValueGradClipper
from equihash.utils.more_torch import kld_loss
from equihash.utils.unique_randint import unique_randint
softplus = torch.nn.Softplus()

from typing import List, Dict
from dataclasses import dataclass, field

@dataclass
class TrainingLog:
    avg_loss: float
    max_saturation_ratio: float
    max_clipping_ratio: float
        
    def __repr__(self):
        return (f'avg_loss:{self.avg_loss:.4f} - '
                f'max_saturation:{100*self.max_saturation_ratio:.4f}% - '
                f'max_clipping:{100*self.max_clipping_ratio:.4f}%')
    
@dataclass
class TrainingLogs:
    losses: List[float] = field(default_factory=list)
    saturation_ratios: List[float] = field(default_factory=list)
    clipping_ratios: List[float] = field(default_factory=list)
    logs: Dict[int, TrainingLog] = field(default_factory=dict)
        
    def reset(self):
        self.losses.clear()
        self.saturation_ratios.clear()
        self.clipping_ratios.clear()
        return self
    
    def aggregate(self, step):
        log = TrainingLog(
            avg_loss=float(np.mean(self.losses)) if self.losses else float('nan'),
            max_saturation_ratio=max(self.saturation_ratios) if self.saturation_ratios else float('nan'),
            max_clipping_ratio=max(self.clipping_ratios) if self.clipping_ratios else float('nan'),
        )
        self.logs[step] = log
        self.reset()
        return log
    
    def state_dict(self):
        return {step:log.__dict__ for step, log in self.logs.items()}
    
    def load_state_dict(self, state):
        for step, log in state.items():
            self.logs[step] = TrainingLog(**log)
        return self
    
    def describe(self, step):
        return f'step:{step} - {self.logs[step]}'
    
class JMLHLoss(torch.nn.Module):
    def __init__(self, nbits, nb_classes, kld_coeff, generator, bias=True, deterministic=False, binarization='sign'):
        if binarization not in ('sign', 'heaviside'):
            raise ValueError(f'binarization must be "sign" or "heaviside", got "{binarization}".')
        super().__init__()
        self.kld_coeff = kld_coeff
        self.generator = generator
        self.deterministic = deterministic
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(nbits, nb_classes, bias=bias),
            torch.nn.LogSoftmax(dim=1),
        )
        self.binarization = straight_through_sign if binarization=='sign' else straight_through_heaviside
        
    def epsilon_like(self, logits):
        if self.deterministic: return 0.5
        return torch.rand(
            logits.shape,
            dtype=logits.dtype,
            device=logits.device,
            generator=self.generator
        )
        
    def forward(self, logits, labels):
        #logits.shape = (bs, nbits)
        #labels.shape = (bs,)
        epsilon = self.epsilon_like(logits)
        codes = self.binarization(torch.sigmoid(logits) - epsilon)
        cls_logp = self.classifier(codes)[torch.arange(len(codes)), labels]
        return -cls_logp.mean() + self.kld_coeff*kld_loss(logits)

class JMLHTrainer:
    def __init__(self, net, loader, nb_classes,
                 loss_kwargs={
                     'kld_coeff':0.1,
                     'bias':True,
                     'deterministic':False,
                     'binarization':'heaviside'},
                 optim_class='Adam', optim_kwargs={'lr': 0.001},
                 clip_value=1, nb_batch_per_step=1):
        self.step = 0
        self.net = net
        self.loader = loader
        self.nb_classes = nb_classes
        self.loss_function = JMLHLoss(self.nbits, nb_classes, generator=self.loader.torch_generator, **loss_kwargs)
        if self.loader.device == 'cuda': self.loss_function.cuda()
        rng = np.random.RandomState(0xcafe)
        self.training_labels = self.loader.generate_labels(n=1, k=self.nb_classes, numpy_generator=rng)[0] #numpy array
        self.optim = torch.optim.__dict__[optim_class](self.parameters, **optim_kwargs)
        self.nb_batch_per_step = nb_batch_per_step
        self.training_log = TrainingLogs()
        self.grad_clipper = ValueGradClipper(self.parameters, clip_value)
            
    @property
    def parameters(self):
        return list(self.net.parameters()) + list(self.loss_function.parameters())
        
    @property
    def nbits(self):
        return self.net.k
        
    def train(self, nb_steps):
        step_target = self.step + nb_steps
        while self.step < step_target:
            self.net.train()
            self.net.zero_grad()
            for _ in range(self.nb_batch_per_step):
                labels_id = self.loader.numpy_generator.randint(0, self.nb_classes, self.loader.size)
                labels = torch.tensor(self.training_labels[labels_id])
                x = self.loader.batch_from_labels(labels)
                out = self.net(x)
                sat = saturation_ratio(out.detach(), threshold=9)
                self.training_log.saturation_ratios.append(sat)
                loss = self.loss_function(out, torch.tensor(labels_id)).mean()
                self.training_log.losses.append(float(loss))
                (loss/self.nb_batch_per_step).backward()

            clip_ratio = self.grad_clipper.clip()
            self.training_log.clipping_ratios.append(clip_ratio)
            self.optim.step()
            self.step += 1
        return self.step
    
    def aggregate(self):
        self.training_log.aggregate(self.step)
        return self
    
    def state_dict(self):
        state = {'step': self.step,
                 'loader': self.loader.get_state(),
                 'optim': self.optim.state_dict(),
                 'training_log': self.training_log.state_dict()}
        return state
    
    def load_state_dict(self, state):
        self.step = state['step']
        self.loader.set_state(state['loader'])
        self.optim.load_state_dict(state['optim'])
        self.training_log.load_state_dict(state['training_log'])
        return self