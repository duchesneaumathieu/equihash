import numpy as np
from dataclasses import dataclass
from scipy.stats import binom
from .inverse import increasing_inverse_infimum, decreasing_inverse_infimum    
from decimal import Decimal, getcontext

def pearson_clopper_lower_bound(n, k, alpha, tol=None):
    if k==0: return 0.
    f = lambda p: 1 - binom.cdf(p=p, n=n, k=k-1) #P(X >= k)
    tol = min(1/1000*n, 1/10**9) if tol is None else tol
    return increasing_inverse_infimum(f, alpha, 0., 1., tol=tol)

def pearson_clopper_upper_bound(n, k, alpha, tol=None):
    if k==n: return 1.
    f = lambda p: binom.cdf(p=p, n=n, k=k) #P(X <= k)
    tol = min(1/1000*n, 1/10**9) if tol is None else tol
    return decreasing_inverse_infimum(f, alpha, 0., 1., tol=tol)

def pearson_clopper_confidence_interval(n, k, alpha, tol=None):
    if k==0: return 0., pearson_clopper_upper_bound(n, k, alpha, tol=tol)
    if k==n: return pearson_clopper_lower_bound(n, k, alpha, tol=tol), 1.
    return pearson_clopper_lower_bound(n, k, alpha/2, tol=tol), pearson_clopper_upper_bound(n, k, alpha/2, tol=tol)

@dataclass
class PearsonClopperEstimate:
    n: int
    k: int
    lambd: float
    
    @property
    def estimate(self):
        return self.k / self.n
    
    def confidence_interval(self):
        return pearson_clopper_confidence_interval(self.n, self.k, 1-self.lambd)
    
    def __repr__(self):
        a, b = self.confidence_interval()
        return f'{self.estimate:.2e} in [{a:.2e}, {b:.2e}] with probability {self.lambd:.2f}'

def variance_upper_bound(n, alpha, x):
    return Decimal(2)/(n**2 - n) * (x + (2*n-4)*(x + alpha)**Decimal('1.5') - (2*n-3)*x**2)

def z_score(n, alpha, x_hat, x):
    return (x_hat - x) / variance_upper_bound(n, alpha, x).sqrt()

def chebyshev_confidence_interval_lower_bound(n, k, alpha, gamma, lambd):
    alpha = Decimal(alpha)
    gamma = Decimal(gamma)
    z = 1 / (1 - Decimal(lambd)).sqrt()
    x_hat = 2*Decimal(k)/(n**2 - n)
    
    a, b = Decimal(0), Decimal(1)
    nb_rounds = int(getcontext().prec * np.log2(10) + 1)
    for i in range(nb_rounds):
        x = (a + b) / 2
        delta = z_score(n, alpha, x_hat, x) - z
        if delta == 0: return x
        elif delta > 0: a = x
        else: b = x
    return a / gamma

def chebyshev_confidence_interval_upper_bound(n, k, alpha, gamma, lambd):
    alpha = Decimal(alpha)
    gamma = Decimal(gamma)
    z = 1 / (1 - Decimal(lambd)).sqrt()
    x_hat = 2*Decimal(k)/(n**2 - n)
    
    a, b = Decimal(0), Decimal(1)
    nb_rounds = int(getcontext().prec * np.log2(10) + 1)
    for i in range(nb_rounds):
        x = (a + b) / 2
        delta = z_score(n, alpha, x_hat, x) + z # + instead of -
        if delta == 0: return x
        elif delta > 0: a = x
        else: b = x
    return b / gamma #return b instead of a

def chebyshev_confidence_interval(n, k, alpha, gamma, lambd):
    """
    Parameters
    ----------
        n: number of samples
        k in \{0, 1, ..., n**2-n\}: the number of collisions 
        alpha = max_l P(L=l)
        gamma = P(L_1 \\neq L_2)
        lambd in (0, 1): confidence interval
    
    Returns
    -------
        a in [0, 1): The Chebyshev collision lower bound
        b in (0, 1]: The Chebyshev collision upper bound
    """
    a = chebyshev_confidence_interval_lower_bound(n, k, alpha, gamma, lambd)
    b = chebyshev_confidence_interval_upper_bound(n, k, alpha, gamma, lambd)
    return a, b

@dataclass
class ChebyshevICREstimate:
    n: int
    k: int
    alpha: float
    gamma: float
    lambd: float
    
    @property
    def estimate(self):
        dissonance_rate = 2 * self.k / (self.n**2 - self.n)
        gamma = Decimal(self.gamma) if isinstance(dissonance_rate, Decimal) else self.gamma
        return dissonance_rate / gamma
    
    def confidence_interval(self):
        return chebyshev_confidence_interval(self.n, self.k, self.alpha, self.gamma, self.lambd)
    
    def __repr__(self):
        a, b = self.confidence_interval()
        return f'{self.estimate:.2e} in [{a:.2e}, {b:.2e}] with probability {self.lambd:.2f}'
