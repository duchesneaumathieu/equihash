import torch
from itertools import combinations
from collections import defaultdict

from typing import List, Tuple, Dict
from dataclasses import dataclass, field

from equihash.losses import BernoulliCombinations
from equihash.utils import ValueGradClipper
from equihash.evaluation import saturation_ratio
from equihash.utils.more_torch import torch_lse, torch_logsumexp
log_sigmoid = torch.nn.LogSigmoid()


def unique_list(seq):
    #https://stackoverflow.com/questions/480214/how-do-i-remove-duplicates-from-a-list-while-preserving-order
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

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
    steps: List[float] = field(default_factory=list)
    avg_shannon_gradient: Dict[Tuple[int, int], List[float]] = field(default_factory=lambda: defaultdict(list))
    avg_hamming_gradient: List[float] = field(default_factory=list)
    max_saturation_ratio: List[float] = field(default_factory=list)
    max_clipping_ratio: List[float] = field(default_factory=list)
    avg_shannon_gradients: Dict[Tuple[int, int], List[float]] = field(default_factory=lambda: defaultdict(list))
    avg_hamming_gradients: List[float] = field(default_factory=list)
    saturation_ratios: List[float] = field(default_factory=list)
    clipping_ratios: List[float] = field(default_factory=list)
        
    def aggregate(self, step):
        self.steps.append(step)
        for k, v in self.avg_shannon_gradients.items():
            mean = sum(v)/len(v)
            self.avg_shannon_gradient[k].append(mean)
        self.avg_hamming_gradient.append(sum(self.avg_hamming_gradients)/len(self.avg_hamming_gradients))
        self.max_saturation_ratio.append(max(self.saturation_ratios))
        self.max_clipping_ratio.append(max(self.clipping_ratios))
        self.reset()
        return self
    
    def reset(self):
        self.avg_shannon_gradients = defaultdict(list)
        self.avg_hamming_gradients = list()
        self.saturation_ratios = list()
        self.clipping_ratios = list()
        return self
        
    def describe(self, k):
        s = ', '.join([f'shannon_grad_{n}c{m}={v[k]:.8f}' for (n,m),v in self.avg_shannon_gradient.items()])
        return (f'step={self.steps[k]}: '
                f'{s}, '
                f'hamming_grad={self.avg_hamming_gradient[k]:.8f}, '
                f'max_saturation={100*self.max_saturation_ratio[k]:.4f}%, '
                f'max_clipping={100*self.max_clipping_ratio[k]:.4f}%')

class ShannonHammingTrainer:
    def __init__(self, net, loader,
                 hamming_lambda, shannon_params, striding=True,
                 shannon_batch_size=None,
                 optim_class='Adam', optim_kwargs={'lr': 0.001},
                 clip_value=1, nb_batch_per_step=1):
        #shannon_params: List[Tuple[int, int, float]]
        #i.e., shannon_params is a list a triplet (n, k, lambda)
        self.step = 0
        self.net = net
        self.loader = loader
        self.training_log = TrainingLog()
        self.nb_batch_per_step = nb_batch_per_step
        self.optim = torch.optim.__dict__[optim_class](net.parameters(), **optim_kwargs)
        self.grad_clipper = ValueGradClipper(net.parameters(), clip_value)
        
        self.nbits = net.k
        self.striding = striding
        self.hamming_lambda = hamming_lambda
        self.shannon_params = shannon_params
        self.shannon_batch_size = shannon_batch_size
        self.shannon_combinations = {(n,k):self.get_bernoulli_combinations(n, k) for n, k, l in shannon_params}
        
    def get_bernoulli_combinations(self, n, k):
        if self.nbits % n != 0:
            raise ValueError(f'{n} must divide the number of bits ({self.nbits})')

        ratio = self.nbits // n
        aligned_groups = [list(range(i*n, (i+1)*n)) for i in range(ratio)]
        strided_groups = [list(range(i, self.nbits, ratio)) for i in range(ratio)]
        all_groups = aligned_groups+strided_groups if self.striding else aligned_groups
        combs = sum([list(combinations(p, k)) for p in all_groups], [])
        return BernoulliCombinations(unique_list(combs))
        
    def shannon_grad_wrt_logits(self, logits, shannon_lambda, shannon_combination):
        #logits.shape = (n, 2, nbits)
        logits = logits.detach()
        logits.requires_grad = True
        logits_view = logits.view(-1, self.nbits)
        ent = shannon_combination.average_distributions_entropy(logits_view, batch_size=self.shannon_batch_size)
        shannon_loss = -shannon_lambda*ent.sum()
        shannon_loss.backward()
        return logits.grad
        
    def hamming_grad_wrt_logits(self, logits):
        #logits.shape = (n, 2, nbits)
        logits = logits.detach()
        logits.requires_grad = True
        equality = multi_bernoulli_equality(logits[:,0], logits[:,1])[1].sum(dim=1)
        hamming_loss = -self.hamming_lambda*equality.mean()
        hamming_loss.backward()
        return logits.grad
    
    def compute_logits(self):
        batches = list()
        logits = list()
        with torch.no_grad():
            for _ in range(self.nb_batch_per_step):
                x = self.loader.positive_batch()
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
            
            logits_grad = self.hamming_grad_wrt_logits(logits)
            self.training_log.avg_hamming_gradients.append(float(logits_grad.abs().mean()))
            for n, k, shannon_lambda in self.shannon_params:
                shannon_grad = self.shannon_grad_wrt_logits(logits, shannon_lambda, self.shannon_combinations[(n,k)])
                self.training_log.avg_shannon_gradients[(n,k)].append(float(shannon_grad.abs().mean()))
                logits_grad += shannon_grad
                
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
                 'training_log': self.training_log.__dict__}
        return state
    
    def load_state_dict(self, state):
        self.step = state['step']
        self.loader.set_state(state['loader'])
        self.optim.load_state_dict(state['optim'])
        self.training_log.__dict__ = state['training_log']
        return self