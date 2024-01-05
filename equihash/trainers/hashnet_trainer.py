import torch
import numpy as np
from equihash.utils import PolynomialStepScheduler, ValueGradClipper
from equihash.evaluation import saturation_ratio
softplus = torch.nn.Softplus()

from typing import List, Dict
from dataclasses import dataclass, field

@dataclass
class TrainingLog:
    beta: float
    avg_loss: float
    max_saturation_ratio: float
    max_clipping_ratio: float
    positive_pairs_grad_quantiles: List[float] = field(default_factory=list)
    negative_pairs_grad_quantiles: List[float] = field(default_factory=list)
    positive_pairs_dots_histogram: List[int] = field(default_factory=list)
    negative_pairs_dots_histogram: List[int] = field(default_factory=list)
        
    def __repr__(self):
        pq = ','.join([f'{q:.4f}' for q in self.positive_pairs_grad_quantiles])
        nq = ','.join([f'{q:.4f}' for q in self.negative_pairs_grad_quantiles])
        return (f'beta:{self.beta:.4f} - '
                f'avg_loss:{self.avg_loss:.4f} - '
                f'max_saturation:{100*self.max_saturation_ratio:.4f}% - '
                f'max_clipping:{100*self.max_clipping_ratio:.4f}% - '
                f'positive_grad_quantiles:[{pq}] - '
                f'negative_grad_quantiles:[{nq}]')

@dataclass
class TrainingLogs:
    losses: List[float] = field(default_factory=list)
    saturation_ratios: List[float] = field(default_factory=list)
    clipping_ratios: List[float] = field(default_factory=list)
    positive_pairs_grads: List[float] = field(default_factory=list)
    negative_pairs_grads: List[float] = field(default_factory=list)
    positive_dots_histograms: List[List[int]] = field(default_factory=list)
    negative_dots_histograms: List[List[int]] = field(default_factory=list)
    logs: Dict[int, TrainingLog] = field(default_factory=dict)
        
    def reset(self):
        self.losses.clear()
        self.saturation_ratios.clear()
        self.clipping_ratios.clear()
        self.positive_pairs_grads.clear()
        self.negative_pairs_grads.clear()
        self.positive_dots_histograms.clear()
        self.negative_dots_histograms.clear()
        return self
    
    def aggregate(self, step, beta):
        pg = self.positive_pairs_grads
        ng = self.negative_pairs_grads
        ph = self.positive_dots_histograms
        nh = self.negative_dots_histograms
        log = TrainingLog(
            beta=beta,
            avg_loss=float(np.mean(self.losses)) if self.losses else float('nan'),
            max_saturation_ratio=max(self.saturation_ratios) if self.saturation_ratios else float('nan'),
            max_clipping_ratio=max(self.clipping_ratios) if self.clipping_ratios else float('nan'),
            positive_pairs_grad_quantiles=np.quantile(pg, [0, .05, .5, .95, 1]).tolist() if pg else None,
            negative_pairs_grad_quantiles=np.quantile(ng, [0, .05, .5, .95, 1]).tolist() if ng else None,
            positive_pairs_dots_histogram=np.array(ph).sum(axis=1).tolist() if ph else None,
            negative_pairs_dots_histogram=np.array(nh).sum(axis=1).tolist() if nh else None,
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
        self.training_log = TrainingLogs()
        self.alpha = 10./self.nbits if alpha is None else alpha
        self.beta_scheduler = PolynomialStepScheduler(**beta_scheduler_kwargs)
        self.grad_clipper = ValueGradClipper(net.parameters(), clip_value)
        
    @property
    def nbits(self):
        return self.net.k
        
    @property
    def beta(self):
        return self.beta_scheduler(self.step)
    
    def positive_loss_function(self, logits):
        dots = torch.tanh(self.beta*logits).prod(dim=1).sum(dim=1)
        losses = softplus(self.alpha*dots) - self.alpha*dots
        if return_dots:
            return losses, dots
        return losses
    
    def negative_loss_function(self, logits):
        dots = torch.tanh(self.beta*logits).prod(dim=1).sum(dim=1)
        losses = softplus(self.alpha*dots)
        if return_dots:
            return losses, dots
        return losses
    
    def loss_function(self, logits, s, return_dots=False):
        dots = torch.tanh(self.beta*logits).prod(dim=1).sum(dim=1)
        losses = softplus(self.alpha*dots) - s*self.alpha*dots
        if return_dots:
            return losses, dots
        return losses
    
    def append_dots_histograms(self, dots, s):
        n = self.nbits
        pos_hist = torch.histc(dots[s], bins=2*n, min=-n, max=n).type(torch.int64).tolist()
        neg_hist = torch.histc(dots[~s], bins=2*n, min=-n, max=n).type(torch.int64).tolist()
        self.training_log.positive_dots_histograms.append(pos_hist)
        self.training_log.negative_dots_histograms.append(neg_hist)
        
    def train(self, nb_steps):
        step_target = self.step + nb_steps
        while self.step < step_target:
            self.net.train()
            self.net.zero_grad()
            positive_pairs_grads = list()
            negative_pairs_grads = list()
            for _ in range(self.nb_batch_per_step):
                x, s = self.loader.mixed_pairs_batch()
                out = self.net(x)
                out.retain_grad()
                sat = saturation_ratio(out.detach(), threshold=9)
                self.training_log.saturation_ratios.append(sat)
                losses, dots = self.loss_function(out, s, return_dots=True)
                self.append_dots_histograms(dots, s)
                loss = losses.mean()
                self.training_log.losses.append(float(loss))
                (loss/self.nb_batch_per_step).backward()
                positive_pairs_grads.append(out.grad[s].sum(dim=0).cpu().numpy())
                negative_pairs_grads.append(out.grad[~s].sum(dim=0).cpu().numpy())

            self.training_log.positive_pairs_grads.append(np.abs(np.array(positive_pairs_grads).sum(axis=0)).sum())
            self.training_log.negative_pairs_grads.append(np.abs(np.array(negative_pairs_grads).sum(axis=0)).sum())
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
                 'training_log': self.training_log.state_dict()}
        return state
    
    def load_state_dict(self, state):
        self.step = state['step']
        self.loader.set_state(state['loader'])
        self.optim.load_state_dict(state['optim'])
        self.training_log.load_state_dict(state['training_log'])
        return self