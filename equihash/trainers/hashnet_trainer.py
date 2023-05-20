import torch
from equihash.losses import HashNetLoss
from equihash.utils import PolynomialStepScheduler, ValueGradClipper
from equihash.evaluation import saturation_ratio

from typing import List
from dataclasses import dataclass, field

@dataclass
class TrainingLog:
    steps: List[float] = field(default_factory=list)
    betas: List[float] = field(default_factory=list)
    avg_loss: List[float] = field(default_factory=list)
    max_saturation_ratio: List[float] = field(default_factory=list)
    max_clipping_ratio: List[float] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    saturation_ratios: List[float] = field(default_factory=list)
    clipping_ratios: List[float] = field(default_factory=list)
        
    def aggregate(self, step, beta):
        self.steps.append(step)
        self.betas.append(beta)
        self.avg_loss.append(sum(self.losses)/len(self.losses) if self.losses else None)
        self.max_saturation_ratio.append(max(self.saturation_ratios) if self.saturation_ratios else None)
        self.max_clipping_ratio.append(max(self.clipping_ratios) if self.clipping_ratios else None)
        self.reset()
        return self
    
    def reset(self):
        self.losses = list()
        self.saturation_ratios = list()
        self.clipping_ratios = list()
        return self
        
    def describe(self, k):
        return (f'step={self.steps[k]} (beta={self.betas[k]:.4f}): '
                f'avg_loss={self.avg_loss[k]:.4f}, '
                f'max_saturation={100*self.max_saturation_ratio[k]:.4f}%, '
                f'max_clipping={100*self.max_clipping_ratio[k]:.4f}%')

class HashNetTrainer:
    def __init__(self, net, loader, alpha=None,
                 beta_scheduler_kwargs={
                     'init': 1.0,
                     'gamma':0.005, 
                     'power':0.5,
                     'step_size':200},
                 optim_class='Adam', optim_kwargs={'lr': 0.001},
                 clip_value=1, nb_batch_per_step=1):
        self.step = 0
        self.net = net
        self.loader = loader
        self.optim = torch.optim.__dict__[optim_class](net.parameters(), **optim_kwargs)
        self.nb_batch_per_step = nb_batch_per_step
        self.training_log = TrainingLog()
        self.alpha = 10./net.k if alpha is None else alpha
        self.loss_function = HashNetLoss(alpha=self.alpha)
        self.beta_scheduler = PolynomialStepScheduler(**beta_scheduler_kwargs)
        self.grad_clipper = ValueGradClipper(net.parameters(), clip_value)
        
    @property
    def beta(self):
        return self.beta_scheduler(self.step)
        
    def train(self, nb_steps):
        step_target = self.step + nb_steps
        while self.step < step_target:
            self.net.train()
            self.net.zero_grad()
            for _ in range(self.nb_batch_per_step):
                x, s = self.loader.batch()
                out = self.net(x)
                sat = saturation_ratio(out.detach(), threshold=9)
                self.training_log.saturation_ratios.append(sat)
                loss = self.loss_function(out, s, beta=self.beta).mean()
                self.training_log.losses.append(float(loss))
                (loss/self.nb_batch_per_step).backward()

            clip_ratio = self.grad_clipper.clip()
            self.training_log.clipping_ratios.append(clip_ratio)
            self.optim.step()
            self.step += 1
        return self.step
    
    def aggregate(self):
        self.training_log.aggregate(self.step, self.beta)
        return self
    
    def state_dict(self):
        state = {'step': self.step,
                 'loader': self.loader.get_state(),
                 'optim': self.optim.state_dict(),
                 'training_log': self.training_log.__dict__}
        return state
    
    def load_state_dict(self, state):
        self.step = state['step']
        self.loader.set_state(state['loader'])
        self.optim.load_state_dict(state['optim'])
        self.training_log.__dict__ = state['training_log']
        return self