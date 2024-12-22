import torch
import numpy as np
from itertools import combinations
from collections import defaultdict

from typing import List, Tuple, Dict
from dataclasses import dataclass, field

from equihash.losses.entropy import TuplesEntropyGradient
from equihash.losses.regularization import HuberLoss
from equihash.utils import ValueGradClipper, covering_random_combinations, add_combinations
from equihash.evaluation import saturation_ratio
from equihash.utils.more_torch import torch_lse, torch_logsumexp, torch_invperm
log_sigmoid = torch.nn.LogSigmoid()


def multi_bernoulli_equality(xz, yz):
    """
    Compute the bitwise log probability that two Multi-Bernoulli are equal.
    
    Parameters
    ----------
    xz : torch.tensor
        the logits (before sigmoid) of the first Multi-Bernoulli
    yz : torch.tensor
        the logits (before sigmoid) of the second Multi-Bernoulli
        
    Returns
    -------
    log_p0 : torch.tensor
        the bitwise log probability that the two Multi-Bernoulli are not equal
    log_p1 : torch.tensor
        the bitwise log probability that the two Multi-Bernoulli are equal
        
    Notes
    -----
    xz and yz need not to have the same shape, but they should
    be broadcastable.
    """
    xp, yp, xn, yn = map(log_sigmoid, (xz, yz, -xz, -yz))
    log_p0 = torch_logsumexp(xp + yn, xn + yp)
    log_p1 = torch_logsumexp(xp + yp, xn + yn)
    return log_p0, log_p1

@dataclass
class TrainingLog:
    max_clipping_ratio: float
    max_saturation_ratio: float
    shannon_gradient_maximum_norm_quantiles: List[float]
    hamming_gradient_maximum_norm_quantiles: List[float]
    huber_gradient_maximum_norm_quantiles: List[float]
    
    def __repr__(self):
        sq = ','.join([f'{q:.4f}' for q in self.shannon_gradient_maximum_norm_quantiles])
        hq = ','.join([f'{q:.4f}' for q in self.hamming_gradient_maximum_norm_quantiles])
        uq = ','.join([f'{q:.6f}' for q in self.huber_gradient_maximum_norm_quantiles])
        return (f'max_saturation:{100*self.max_saturation_ratio:.4f}% - '
                f'max_clipping:{100*self.max_clipping_ratio:.4f}% - '
                f'shannon_grad_quantiles:[{sq}] - '
                f'hamming_grad_quantiles:[{hq}] - '
                f'huber_grad_mquantiles:[{uq}]')
    
@dataclass
class TrainingLogs:
    saturation_ratios: List[float] = field(default_factory=list)
    clipping_ratios: List[float] = field(default_factory=list)
    shannon_gradient_maximum_norms: List[float] = field(default_factory=list)
    hamming_gradient_maximum_norms: List[float] = field(default_factory=list)
    huber_gradient_maximum_norms: List[float] = field(default_factory=list)
    logs: Dict[int, TrainingLog] = field(default_factory=dict)
        
    def reset(self):
        self.saturation_ratios.clear()
        self.clipping_ratios.clear()
        self.shannon_gradient_maximum_norms.clear()
        self.hamming_gradient_maximum_norms.clear()
        self.huber_gradient_maximum_norms.clear()
        return self
    
    def aggregate(self, step):
        nans_quantiles = 5*[float('nan')]
        sm = self.shannon_gradient_maximum_norms
        hm = self.hamming_gradient_maximum_norms
        um = self.huber_gradient_maximum_norms
        log = TrainingLog(
            max_saturation_ratio=max(self.saturation_ratios) if self.saturation_ratios else float('nan'),
            max_clipping_ratio=max(self.clipping_ratios) if self.clipping_ratios else float('nan'),
            shannon_gradient_maximum_norm_quantiles=np.quantile(sm, [0, .05, .5, .95, 1]).tolist() if sm else nans_quantiles,
            hamming_gradient_maximum_norm_quantiles=np.quantile(hm, [0, .05, .5, .95, 1]).tolist() if hm else nans_quantiles,
            huber_gradient_maximum_norm_quantiles=np.quantile(um, [0, .05, .5, .95, 1]).tolist() if um else nans_quantiles,
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

class ShannonHammingTrainer:
    def __init__(self, net, loader,
                 shannon_lambda, hamming_lambda,
                 shannon_gradient_kwargs = {
                     'batch_size': 512,
                     'gradient_type': 'normal',
                     'prior': 1.,
                 },
                 shannon_tuple_size=3,
                 shannon_tuple_cover_size=None,
                 shannon_minimum_number_of_tuples=1,
                 shannon_shuffle_logits=False,
                 huber_loss_kwargs = {
                     "maximum_slope": 0.0,
                     "branching_point": 9.0,
                 },
                 optim_class='Adam', optim_kwargs={'lr': 0.001},
                 clip_value=1, nb_batch_per_step=1):
        shannon_tuple_cover_size = shannon_tuple_size if shannon_tuple_cover_size is None else shannon_tuple_cover_size
        
        self.step = 0
        self.net = net
        self.loader = loader
        self.training_log = TrainingLogs()
        self.nb_batch_per_step = nb_batch_per_step
        self.optim_kwargs = optim_kwargs
        self.optim = torch.optim.__dict__[optim_class](net.parameters(), **optim_kwargs)
        self.grad_clipper = ValueGradClipper(net.parameters(), clip_value)
        
        self.shannon_lambda = shannon_lambda
        self.hamming_lambda = hamming_lambda
        self.huber_loss = HuberLoss(**huber_loss_kwargs)
        self.shannon_shuffle_logits = shannon_shuffle_logits
        tuples_rng = np.random.RandomState(0xcafe)
        tuples = covering_random_combinations(n=self.nbits, k=shannon_tuple_size, cover_k=shannon_tuple_cover_size, rng=tuples_rng)
        tuples = add_combinations(n=self.nbits, k=shannon_tuple_size, combs=tuples, target_size=shannon_minimum_number_of_tuples, rng=tuples_rng)
        self.shannon_tuple_gradient = TuplesEntropyGradient(tuples, **shannon_gradient_kwargs)
        
    @property
    def nbits(self):
        return self.net.k
        
    def shannon_grad_wrt_logits(self, logits):
        logits = logits.detach()
        n, _, nbits = logits.shape #logits.shape = (n, 2, nbits)
        logits_view = logits.view(2*n, nbits)
        
        if self.shannon_shuffle_logits:
            #using loader.numpy_generator to avoid managing another rng
            perm = torch.tensor(self.loader.numpy_generator.permutation(nbits))
            logits_view = logits_view[:, perm]
            
        logits_view_grad = self.shannon_tuple_gradient.negative_logits_gradient(logits_view)
        
        if self.shannon_shuffle_logits:
            logits_view_grad = logits_view_grad[:, torch_invperm(perm)]
            
        return self.shannon_lambda * logits_view_grad.view(n, 2, nbits)
        
    def hamming_grad_wrt_logits(self, logits):
        #logits.shape = (n, 2, nbits)
        logits = logits.detach()
        logits.requires_grad = True
        equality = multi_bernoulli_equality(logits[:,0], logits[:,1])[1].sum(dim=1)
        hamming_loss = -self.hamming_lambda*equality.mean()
        hamming_loss.backward()
        return logits.grad
    
    def huber_grad_wrt_logits(self, logits):
        #logits.shape = (n, 2, nbits)
        logits = logits.detach()
        logits.requires_grad = True
        huber_loss = self.huber_loss(logits).sum(dim=2).mean()
        huber_loss.backward()
        return logits.grad
    
    def compute_logits(self):
        batches = list()
        logits = list()
        with torch.no_grad():
            for _ in range(self.nb_batch_per_step):
                x = self.loader.positive_pairs_batch()
                out = self.net(x)
                sat = saturation_ratio(out.detach(), threshold=9)
                self.training_log.saturation_ratios.append(sat)
                batches.append(x)
                logits.append(out)
        return batches, torch.cat(logits, dim=0)
        
    def train(self, nb_steps):
        step_target = self.step + nb_steps
        while self.step < step_target:
            self.net.train()
            self.net.zero_grad()
            batches, logits = self.compute_logits()
            
            shannon_grad = self.shannon_grad_wrt_logits(logits)
            hamming_grad = self.hamming_grad_wrt_logits(logits)
            huber_grad = self.huber_grad_wrt_logits(logits)
            logits_grad = shannon_grad + hamming_grad + huber_grad
            
            self.training_log.shannon_gradient_maximum_norms.append(float(shannon_grad.abs().sum(dim=(1,2)).mean()))
            self.training_log.hamming_gradient_maximum_norms.append(float(hamming_grad.abs().sum(dim=(1,2)).mean()))
            self.training_log.huber_gradient_maximum_norms.append(float(huber_grad.abs().sum(dim=(1,2)).mean()))
                
            i = 0
            for x in batches:
                grad = logits_grad[i: i+len(x)]
                self.net(x).backward(grad)
                i += len(x)

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
        for group in state['optim']['param_groups']:
            #update learning rate (and more) if changed
            group.update(self.optim_kwargs)
        self.optim.load_state_dict(state['optim'])
        self.training_log.load_state_dict(state['training_log'])
        return self