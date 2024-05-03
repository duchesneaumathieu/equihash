def increasing_inverse_infimum(f, y, min_x, max_x, tol=1e-12, limit=1000, raise_on_limit=True):
    #given an increasing function f
    #find x in (min_x, max_x) s.t. y - tol <= f(x) <= y
    n = 0
    while n < limit:
        x = (min_x + max_x)/2
        delta = f(x) - y
        if delta > 0:
            #f(x) is too big, x* must be smaller than x
            max_x = x
        elif -delta <= tol:
            #y-tol <= f(x) <= y
            return x
        else: # f(x) < y-tol
            min_x = x
        n += 1
    #n == limit if we are here
    if raise_on_limit:
        raise RuntimeError(f'limit ({limit:,}) reached.')
    return min_x #return x s.t. f(x) < y-tol < y

def increasing_inverse_supremum(f, y, min_x, max_x, tol=1e-12, limit=1000, raise_on_limit=True):
    #given an increasing function f
    #find x in (min_x, max_x) s.t. y <= f(x) <= y + tol
    #which holds iff y - tol <= f(x) - tol <= y
    #so it is the infimum of the function f-tol
    return increasing_inverse_infimum(lambda x: f(x)-tol, y, min_x, max_x, tol=tol, limit=limit, raise_on_limit=raise_on_limit)

def decreasing_inverse_supremum(f, y, min_x, max_x, tol=1e-12, limit=1000, raise_on_limit=True):
    #given an decreasing function f
    #find x in (min_x, max_x) s.t. y <= f(x) <= y + tol
    #which holds iff -y - tol <= -f(x) <= -y (where -f is now an increasing function)
    #so it is the inverse infimum of the function -f (w.r.t. -y)
    return increasing_inverse_infimum(lambda x: -f(x), -y, min_x, max_x, tol=tol, limit=limit, raise_on_limit=raise_on_limit)

def decreasing_inverse_infimum(f, y, min_x, max_x, tol=1e-12, limit=1000, raise_on_limit=True):
    #given an decreasing function f
    #find x in (min_x, max_x) s.t. y-tol <= f(x) <= y
    #which holds iff -y <= -f(x) <= -y+tol (where -f-tol is now an increasing function)
    #which holds iff -y-tol <= -f(x)-tol <= -y
    #so it is the inverse infimum of the function -f-tol (w.r.t. -y)
    return increasing_inverse_infimum(lambda x: -f(x)-tol, -y, min_x, max_x, tol=tol, limit=limit, raise_on_limit=raise_on_limit)