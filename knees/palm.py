# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 11:31:57 2020

@author: marga
"""
import numpy as np
from odl.solvers import L2NormSquared as odl_l2sq


class PALM():

    def __init__(self, f, g, ud_vars=None, x=None, niter=None, 
                 callback=None, L=None, tol=None):

        if x is None:
            x = f.domain.zero()
            
        if L is None:
            L = [1] * len(x)          
            
        if ud_vars is None:
            ud_vars = list(range(len(x)))
            
        self.ud_vars = ud_vars    
        self.x = x
        self.f = f
        self.g = g
        self.etas = [0.5, 10]
        self.L = L
        self.tol = tol
        self.callback = callback
        self.niter = 0
            
        self.dx = None
        self.x_old = None
        self.x_old2 = None

        self.g_old = None
        self.f_old = None

        if niter is not None:
            self.run(niter)

    def update_coordinate(self, i):
        
        #import time
        #t = time.time()
        
        x = self.x
        x_old = self.x_old
        f = self.f
        g = self.g
        L = self.L
        
        BTsuccess = False
        if i==0:
            bt_vec = range(60)
        else:
            bt_vec = range(60)
        
        
        #gx = g[i](self.x_old[i])
        
        l2sq = odl_l2sq(x[i].space)
        df = f.gradient(x_old)[i]
        #fx = f(self.x_old)
        
        
        #print(i, time.time()-t)
            
        for bt in bt_vec:    #backtracking loop
            
            try:
                g[i].proximal(1/L[i])(x_old[i] - 1/L[i] * df, out=x[i])
            except:
                self.ud_vars.remove(i) #Margaret added this
                break
            # backtracking on Lipschitz constants
            f_new = f(x)
            LHS1 = f_new
            
            self.dx[i] = x[i] - x_old[i]
            
            df_dxi = df.inner(self.dx[i])
            dxi_sq = l2sq(self.dx[i])
            
            RHS1 = self.f_old + df_dxi + L[i]/2 * dxi_sq   
            
            eps = 0e-4
            if LHS1 <= RHS1 + eps:
                x_old[i][:] = x[i]
                self.f_old = f_new
                L[i] *= self.etas[0]
                BTsuccess = True
                break
            
            #            #print(i, bt, LHS1 - RHS1)
            #            if LHS1 > RHS1 + eps:
            #                L[i] *= self.etas[1]
            #                continue
            

            # proximal backtracking
            #            gi_new = g[i](x[i])
            #           LHS2 = gi_new
            #          RHS2 = self.g_old[i] - df_dxi - L[i]/2 * dxi_sq
            
            
            #print(i, bt, LHS2 - RHS2)
            #          if LHS2 <= RHS2 + eps:                
            #              x_old[i][:] = x[i]
            #              self.f_old = f_new
            #              self.g_old[i] = gi_new
            #              L[i] *= self.etas[0]
            #              BTsuccess = True
            #              break
            
            L[i] *= self.etas[1]    
            
            
        # print(i, time.time()-t)
        
        if BTsuccess is False:
            print('No step size found for variable {} after {} backtracking steps'.format(i, bt))
   
        #print('Diff{}:{:3.2e}, bt:{}'.format(i, reldiff, bt))

        if self.tol is not None:
            reldiff = dxi_sq/max(l2sq(x[i]), 1e-4)
            
            #print(reldiff)
        
            if reldiff < self.tol:
                self.ud_vars.remove(i)
                print('Variable {} stopped updating'.format(i))

        
    def update(self):
        self.niter += 1
        
        if self.dx is None:
            self.dx = self.x.copy()
        if self.x_old is None:
            self.x_old = self.x.copy()
        if self.f_old is None:
            self.f_old = self.f(self.x_old)
        if self.g_old is None:
            self.g_old = [self.g[j](self.x_old[j]) for j in range(len(self.x))]
        
        #import time
        
        for i in self.ud_vars:     #loop over variables
            
            #t = time.time()
            self.update_coordinate(i)
            #print(i, time.time()-t)
            

    def run(self, niter=1):
        if self.tol is not None:
            if self.x_old2 is None:
                self.x_old2 = self.x.copy()
            l2sq = odl_l2sq(self.x.space)
            
        for k in range(niter):
            if self.x_old2 is None:
                self.x_old2 = self.x.copy()
            self.x_old2[:] = self.x
            self.update()
            
            dx = []
            for i in range(len(self.x)):
                 l2sq = odl_l2sq(self.x[i].space)   
                 dx.append(l2sq(self.dx[i])/max(l2sq(self.x[i]), 1e-4))
                 
            s = 'obj:{:3.2e}, f:{:3.2e}, g:{:3.2e}, diff:' + '{:3.2e} ' * len(self.x) + 'lip:' + '{:3.2e} ' * len(self.x)
            #print((self.f + self.g)(self.x))
            #print(dx)
           # print(s)
           # print(self.L)
            fx = self.f(self.x)
            gx=0
            for i in self.ud_vars:
                gx+=self.g[i](self.x[i])
            print(s.format(fx + gx, fx, gx, *dx, *self.L))

            if self.callback is not None:
                self.callback(self.x)

            if self.tol is not None:
                l2sq = odl_l2sq(self.x.space)   
                norm = l2sq(self.x_old2)
                if k > 1 and norm > 0:
                    crit = l2sq(self.x_old2-self.x)/norm
                else:
                    crit = np.inf
                    
                if crit < self.tol:
                    print('Stopped iterations with rel. diff. ',crit)
                    break
                else:
#                    print('Relative Difference: ',crit)
                    self.x_old2[:] = self.x

        return self.x

