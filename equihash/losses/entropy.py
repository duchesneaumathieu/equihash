import torch
import numpy as np
from equihash.utils import iterslice
from equihash.utils.more_torch import torch_lse, torch_logsumexp
log_sigmoid = torch.nn.LogSigmoid()

def unique_slice(tuples, start=None, end=None, step=None):
    unique = set()
    tuples = [t[start:end:step] for t in tuples]
    return [t for t in tuples if t not in unique and unique.add(t) is None]

def inverted_getitem(tuples):
    return {tuple(t):n for n, t in enumerate(tuples)}

def sub_index(tuples, sub_tuples, start=None, end=None, step=None):
    inv = inverted_getitem(sub_tuples)
    return [inv[t[start:end:step]] for t in tuples]

def raise_when_variant_lengths(items):
    min_length = min(map(len, items))
    max_length = max(map(len, items))
    if min_length != max_length:
        raise ValueError(f'All items must be the same size, found sizes ranging from {min_length} to {max_length}.')
    return min_length

class PartialOuterProduct:
    def __init__(self, subsets, operator=torch.mul):
        """
        Parameters
        ----------
        subsets : list[tuple[int]]
            The subsets of vectors to compute the outer product. All subsets must be the same size.
            
        operator : Callable[[torch.Tensor, torch.Tensor], torch.Tensor], default=torch.mul
            This allow the user to specify another binary operator. For example, it can be helpful
            to perform the outer product in log space. In which case, operator can be set to torch.add.
            
        Notes
        -----
        This algorithm does not assume any proprety of the "operator". To compute the outer product of v, w, and z,
        it computes (v*w)*z. If another subset starts with (v, w) the computation will be shared. However, if another
        subset ends with (w, z) there will be no obtimization. By supposing associativity we might find a more
        efficient order (other than left to right) of "multiplication". By assuming commutativity, it would be possible
        to be even more efficient. Finding the optimal order of multiplication given the subsets seems quite hard.
        
        Anyway, the computation will be more efficient if many subsets share the same start.
        """
        subsets = list(subsets)
        if not subsets:
            raise ValueError('No subset provided.')
        
        self.k = raise_when_variant_lengths(subsets)
        self.subsets = subsets
        self.operator = operator
        
        if self.k == 0:
            raise ValueError('Empty subsets.')
        elif self.k == 1:
            self.indexes = torch.tensor([s[0] for s in subsets])
        else:
            head_tuples = unique_slice(subsets, start=0, end=2)
            self.head_indexes = {2: torch.tensor([s[0] for s in head_tuples])}
            self.tail_indexes = {2: torch.tensor([s[1] for s in head_tuples])}
            self.build_order = {2: head_tuples}

            for i in range(3, self.k+1):
                new_head_tuples = unique_slice(subsets, start=0, end=i)
                self.head_indexes[i] = torch.tensor(sub_index(new_head_tuples, head_tuples, start=0, end=i-1))
                self.tail_indexes[i] = torch.tensor([s[i-1] for s in new_head_tuples])
                self.build_order[i] = head_tuples = new_head_tuples
        
    def __call__(self, vectors):
        """
        Parameters
        ----------
        vectors : torch.tensor, vector.shape=(*sh, n, d)
        
        Returns
        -------
        outer : torch.tensor, vector.shape=(*sh, N, d**k)
            outer[..., i, j_1*d**(k-1) + j_2*d**(k-2) + ... + j_k] = vectors[..., a_1, j_1]*...*vectors[..., a_k, j_k]
            with (a_1, a_2, ..., a_k) = subsets[i] and N = len(subsets)
        """
        if self.k == 1:
            return vectors[:, self.indexes]
        prev = vectors
        *sh, n, d = vectors.shape
        for i in range(2, self.k+1):
            head_index = self.head_indexes[i]
            tail_index = self.tail_indexes[i]
            prev = self.operator(prev[..., head_index, :, None], vectors[..., tail_index, None, :])
            prev = prev.view(*sh, len(self.build_order[i]), d**i)
        return prev
        
def entropy_from_log_probabilities(log_p):
    # entropy taken along the last dim (i.e., p.exp().sum(dim=-1) must be 1)
    return -np.log2(np.e)*(log_p.exp()*log_p).sum(dim=-1)

def entropy_from_probabilities(p):
    # entropy taken along the last dim (i.e., p.sum(dim=-1) must be 1)
    p = torch.where(p==0, torch.ones_like(p), p) #convert p=0 to p=1 to avoid log(0)
    return -(p * torch.log2(p)).sum(dim=-1)
    
class BitsCombinations:
    def __init__(self, subsets):
        self.outer_product = PartialOuterProduct(subsets, operator=torch.logical_and)
        
    def log_distributions(self, codes):
        #this will returns many -inf
        return self.distributions(codes).log()
    
    def distributions(self, codes):
        #returns onehots
        base_case = torch.stack([~codes, codes], dim=-1)
        return self.outer_product(base_case)
    
    def log_average_distributions(self, codes, batch_size=None):
        #can return -inf
        return self.average_distributions(codes, batch_size=batch_size).log()
    
    def average_distributions(self, codes, batch_size=None):
        N = len(codes)
        for i, (beg,end) in enumerate(iterslice(stop=N, slice_size=batch_size)):
            batch_sum_dists = self.distributions(codes[beg:end]).float().sum(dim=0)
            sum_dists = batch_sum_dists if i==0 else sum_dists + batch_sum_dists
        return sum_dists / N
    
    def average_distributions_entropy(self, codes, batch_size=None):
        average_distributions = self.average_distributions(codes, batch_size=batch_size)
        return entropy_from_probabilities(average_distributions)
    

class BernoulliCombinations:
    def __init__(self, subsets):
        self.outer_product = PartialOuterProduct(subsets, operator=torch.add)
        
    def log_distributions(self, logits):
        #distribution (in log prob) of combinations of size 1 (base case) shape=(bs, n choose 1, 2**1) = (bs, n, 2)
        base_case = torch.stack([log_sigmoid(-logits), log_sigmoid(logits)], dim=-1)
        return self.outer_product(base_case)
    
    def distributions(self, logits):
        return self.log_distributions(logits).exp()
    
    def log_average_distributions(self, logits, batch_size=None):
        N = len(logits)
        for i, (beg,end) in enumerate(iterslice(stop=N, slice_size=batch_size)):
            batch_log_sum_dists = torch_lse(self.log_distributions(logits[beg:end]), dim=0)
            log_sum_dists = batch_log_sum_dists if i==0 else torch_logsumexp(log_sum_dists, batch_log_sum_dists)
        return log_sum_dists - np.log(N)
    
    def average_distributions(self, logits, batch_size=None):
        return self.average_log_distributions(logits, batch_size=batch_size).exp()
    
    def average_distributions_entropy(self, logits, batch_size=None):
        average_log_distributions = self.log_average_distributions(logits, batch_size=batch_size)
        return entropy_from_log_probabilities(average_log_distributions)
    
def yield_sub_tuple(t):
    #e.g. (1,2,3,4) yields (1, (2,3,4)), (2, (1,3,4)), (3, (1,2,4)), (4, (1,2,3))
    for i, j in enumerate(t):
        yield j, t[:i]+t[i+1:]

def unique_subtuples(tuples):
    unique = set()
    tuples = [subtup for tup in tuples for _, subtup in yield_sub_tuple(tup)]
    return [t for t in tuples if t not in unique and unique.add(t) is None]

def _build_relevant_sub_tuples_index(tuples, sub_tuples):
    #Returns a dict that maps bit_index to list of sub_tuple_index
    ii = {t:n for n, t in enumerate(sub_tuples)}
    relevant_sub_tuple_index = dict()
    for tup in tuples:
        for bit_index, relevant_sub_tuple in yield_sub_tuple(tup):
            if bit_index in relevant_sub_tuple_index:
                relevant_sub_tuple_index[bit_index].append(ii[relevant_sub_tuple])
            else: relevant_sub_tuple_index[bit_index] = [ii[relevant_sub_tuple]]
    return {i:torch.tensor(index) for i, index in relevant_sub_tuple_index.items()}

class TuplesEntropyGradient:
    def __init__(self, tuples, batch_size=None, gradient_type='normal', prior=1.):
        self.valid_gradient_types = ('normal', 'pseudo-smooth', 'straight-through')
        if gradient_type not in self.valid_gradient_types:
            raise ValueError(f'gradient_type must be in {self.valid_gradient_types}, got {gradient_type}')
        self.batch_size = batch_size
        self.gradient_type = gradient_type
        self.prior = prior
        
        self.tuples = [tuple(t) for t in tuples]
        self.tuples_size = raise_when_variant_lengths(tuples)
        self.sub_tuples = unique_subtuples(tuples)
        self.relevant_sub_tuples_index = _build_relevant_sub_tuples_index(tuples, self.sub_tuples)
        if self.tuples_size == 1:
            raise NotImplementedError(f'tuples_size must be greater than one (got {self.tuples_size}).')
        self.bernoulli_combinations = BernoulliCombinations(self.sub_tuples)
        self.bits_combinations = BitsCombinations(self.sub_tuples)
        
    def compute_conditional_logits(self, logits, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        log_p0, log_p1 = log_sigmoid(-logits), log_sigmoid(logits)
        
        #init cumulatives
        log_positives_joint_cumuls = dict()
        log_negatives_joint_cumuls = dict()
        
        #each batch adds to the cumulatives
        for batch_id, (beg, end) in enumerate(iterslice(stop=len(logits), slice_size=batch_size)):
            batch = logits[beg:end]
            log_sub_joint_dists= self.bernoulli_combinations.log_distributions(batch)
            for i, index in self.relevant_sub_tuples_index.items():
                log_sub_joint_dists_i = log_sub_joint_dists[:, index] #shape = (bs, len(index), 2**(self.tuples_size-1))
                log_positives_joint_cumuls_batch = torch_lse(log_p1[beg:end,i,None,None] + log_sub_joint_dists_i, dim=0)
                log_negatives_joint_cumuls_batch = torch_lse(log_p0[beg:end,i,None,None] + log_sub_joint_dists_i, dim=0)
                if batch_id==0:
                    log_positives_joint_cumuls[i] = log_positives_joint_cumuls_batch
                    log_negatives_joint_cumuls[i] = log_negatives_joint_cumuls_batch
                    continue
                log_positives_joint_cumuls[i] = torch_logsumexp(log_positives_joint_cumuls[i], log_positives_joint_cumuls_batch)
                log_negatives_joint_cumuls[i] = torch_logsumexp(log_negatives_joint_cumuls[i], log_negatives_joint_cumuls_batch)
                
        #finalize
        conditional_logits = dict()
        for i, index in self.relevant_sub_tuples_index.items():
            #the log(len(logits)) cancels out so no need to normalize
            conditional_logits[i] = log_positives_joint_cumuls[i] - log_negatives_joint_cumuls[i] 
        return conditional_logits
        
    def compute_discrete_conditional_logits(self, logits, batch_size=None, prior=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        prior = self.prior if prior is None else prior
        
        codes = 0 < logits
        N, n = codes.shape
        prior = float(prior)
        
        #init cumulatives
        positives_joint_cumulatives = dict()
        negatives_joint_cumulatives = dict()
        for i, index in self.relevant_sub_tuples_index.items():
            shape = (len(index), 2**(self.tuples_size-1))
            positives_joint_cumulatives[i] = torch.zeros(shape, device=logits.device, dtype=logits.dtype)
            negatives_joint_cumulatives[i] = torch.zeros(shape, device=logits.device, dtype=logits.dtype)
        
        #each batch adds to the cumulatives
        for beg, end in iterslice(stop=N, slice_size=batch_size):
            batch = codes[beg:end]
            sub_joint_dists = self.bits_combinations.distributions(batch)
            for i, index in self.relevant_sub_tuples_index.items():
                sub_joint_dists_i = sub_joint_dists[:, index] #shape = (bs, len(index), 2**(self.tuples_size-1))
                positives_joint_cumulatives[i] += torch.logical_and(batch[:,i,None,None], sub_joint_dists_i).sum(dim=0)
                negatives_joint_cumulatives[i] += torch.logical_and(~batch[:,i,None,None], sub_joint_dists_i).sum(dim=0)
        
        #finalize
        conditional_logits = dict()
        denominator = len(logits) + prior*2**self.tuples_size
        for i, index in self.relevant_sub_tuples_index.items():
            pos_dist = (positives_joint_cumulatives[i] + prior) / denominator
            neg_dist = (negatives_joint_cumulatives[i] + prior) / denominator
            conditional_logits[i] = pos_dist.log() - neg_dist.log()
        return conditional_logits
    
    def negative_logits_gradient(self, logits, batch_size=None, gradient_type=None, prior=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        gradient_type = self.gradient_type if gradient_type is None else gradient_type
        prior = self.prior if prior is None else prior
        
        prob_grad = torch.zeros_like(logits)
        if gradient_type == 'normal':
            conditional_logits = self.compute_discrete_conditional_logits(logits, batch_size=batch_size, prior=prior)
        elif gradient_type == 'pseudo-smooth':
            conditional_logits = self.compute_conditional_logits(logits, batch_size=batch_size)
        elif gradient_type == 'straight-through':
            raise NotImplementedError('gradient_type=="straight-through" is not implemented.')
        else: raise ValueError(f'gradient_type must be in {self.valid_gradient_types}, got {gradient_type}.')
        
        for beg, end in iterslice(stop=len(logits), slice_size=batch_size):
            batch = logits[beg:end]
            sub_joint_dists = self.bernoulli_combinations.distributions(batch)
            for i, index in self.relevant_sub_tuples_index.items():
                sub_joint_dists_i = sub_joint_dists[:, index]
                prob_grad[beg:end, i] = (sub_joint_dists_i*conditional_logits[i]).sum(dim=(1,2))
        
        sigmoid_grad = log_sigmoid(-logits).exp() * log_sigmoid(logits).exp()
        logits_grad = prob_grad * sigmoid_grad / len(logits)
        return logits_grad
