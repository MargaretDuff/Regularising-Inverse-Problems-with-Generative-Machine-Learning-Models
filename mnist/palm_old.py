from odl.solvers import (L2NormSquared, pdhg_stepsize, 
                         proximal_convex_conj_l1, 
                         proximal_convex_conj_l1_l2)
import numpy as np
from odl.operator.tensor_ops import PointwiseInner


class PALM():

    def __init__(self, f, g, ud_vars=None, x=None, niter=None, 
                 callback=None, L=None, TOL=1e-7):

         
        if x is None:
            x = f.domain.zero()
        self.M = len(x.space)       
        if L is None:
            L = 1e2*np.ones(self.M)            
            
        if ud_vars is None:
            ud_vars = list(range(self.M))
            
        self.ud_vars = ud_vars    
        self.x = x
        self.f = f
        self.g = g
        self.theta = 1.1
        self.eta = 1.1
        self.L = L
        self.TOL = TOL

        if callback is not None:
            self.callback = callback

        if niter is not None:
            self.run(niter)

    def update(self):
        x = self.x
        f = self.f
        g = self.g
        eta = self.eta
        theta = self.theta
        L = self.L
        xold = x.copy()
        
        if self.ud_vars:
            for i in self.ud_vars:     #loop over variables
                BTsuccess = False
                if i==0:
                    bt_vec = range(60)
                else:
                    bt_vec = range(60)
                
                eps = 0e-4
                
                regold = g[i](x[i])
                
                for j in bt_vec:    #backtracking loop
                    ss = 1/(theta * L[i])
                    df = f.gradient(x)[i]
    
                    tmpxi = g[i].proximal(ss)(x[i] - ss * df)
                    tmpx = x.copy()
                    tmpx[i] = tmpxi
                    
                    # backtracking on Lipschitz constants
                    LHS1 = f(tmpx)
                    RHS1 = f(xold) + df.inner(tmpxi-xold[i]) + L[i]/2 * L2NormSquared(tmpxi.space)(tmpxi-xold[i])
                    
                    
                    if LHS1 > RHS1 + eps:
                        L[i] *= eta
                        continue
                    
    
                    # proximal backtracking
                    reg = g[i](tmpxi)
                    LHS2 = reg
                    RHS2 = regold - df.inner(tmpxi-xold[i]) - 1/(2*ss) * L2NormSquared(tmpxi.space)(tmpxi-xold[i])
                    
                    if LHS2 <= RHS2 + eps:
                        L[i] /= eta
                        BTsuccess = True
                        break
                                            
                if BTsuccess is False:
                    print('No step size found for {}th variable after {} backtracking steps'.format(i,j))
                
                x[i] = tmpxi            
                l2sq = L2NormSquared(x[i].space)
                try:
                    reldiff = l2sq(x[i]-xold[i])/l2sq(x[i])
                except:
                    reldiff=float("nan")
                print('Difference in {}th variable: {}'.format(i,reldiff))
    
                # if reldiff < self.TOL:
                #     self.ud_vars.remove(i)
                #     print('Variable {} stopped updating'.format(i))
                    
                xold[i] = x[i].copy()

    def run(self, niter=1):
        if self.TOL is not None:
            xm = self.x.copy()
            l2sq = L2NormSquared(xm.space)
            
        for ind in range(niter):
            self.update()

            if self.callback is not None:
                self.callback(self.x)

            if self.TOL is not None:
                if ind>1 and l2sq(xm)>0:
                    crit = l2sq(xm-self.x)/l2sq(xm)
                else:
                    crit = np.inf
                    
                if crit<self.TOL:
                    print('Stopped iterations with rel. diff. ',crit)
                    break
                else:
#                    print('Relative Difference: ',crit)
                    xm = self.x.copy()

        return self.x


def fgp_dual(p, data, sigma, niter, grad, proj_C, proj_P, tol=None, **kwargs):
    callback = kwargs.pop('callback', None)
    if callback is not None and not callable(callback):
        raise TypeError('`callback` {} is not callable'.format(callback))

    factr = 1 / (grad.norm**2 * sigma)

    q = p.copy()
    x = data.space.zero()

    t = 1.

    if tol is None:
        def convergence_eval(p1, p2, k):
            return False
    else:
        def convergence_eval(p1, p2, k):
            return k > 5 and (p1 - p2).norm() / max(p1.norm(), 1) < tol

    pnew = p.copy()

    if callback is not None:
        callback(p)

    for k in range(niter):
        t0 = t
        grad.adjoint(q, out=x)
        proj_C(data - sigma * x, out=x)
        pnew = grad(x, out=pnew)
        pnew *= factr
        pnew += q

        proj_P(pnew, out=pnew)

        converged = convergence_eval(p, pnew, k)

        if not converged:
            # update step size
            t = (1 + np.sqrt(1 + 4 * t0 ** 2)) / 2.

            # calculate next iterate
            q[:] = pnew + (t0 - 1) / t * (pnew - p)

        p[:] = pnew

        if converged:
            t = None
            break

        if callback is not None:
            callback(p)

    # get current image estimate
    x = proj_C(data - sigma * grad.adjoint(p))

    return x
            
            
        
            
            
    
    
    
        
   
