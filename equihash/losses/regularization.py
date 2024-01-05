import torch

class HuberLoss(object):
    """
    Functional for of Huber loss from which we can decide the final slope and the branching point
    between the quadratic and the linear branches. It is equivalent to the HuberLoss from the pytorch
    package (after multipling by a constant) but it allows the user to controls the derivative without
    the need of carefully choosing the right multipling factor.
    
    Parameters
    -----------
    maximum_slope : float
        The the maximum slope of the function, it is the slope of the linear branch.
    branching_point : float
        Below this value, we use a quadratique equation otherwise a linear (affine to be exact) equation.
    """
    def __init__(self, maximum_slope, branching_point):
        self.m = maximum_slope
        self.k = branching_point
        self.c = self.m/(2*self.k)
        self.b = self.c*self.k**2 - self.m*self.k
        
    def __call__(self, x):
        xabs = x.abs()
        return torch.where(xabs<self.k, self.c*x**2, self.m*xabs + self.b)