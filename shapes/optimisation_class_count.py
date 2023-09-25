# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 13:42:41 2020

@author: marga
"""
import numpy as np
import odl 

import odl.contrib.tensorflow
import odl.solvers
import palm_count2 as palm 
import skimage.measure


class optimisation():
    def __init__(self,inv_prob, generative_model):
        self.inv_prob=inv_prob
        self.generative_model=generative_model
        self.x_space=self.inv_prob.x_space
        self.y_space=self.inv_prob.y_space
        self.z_space=self.generative_model.z_space
        self.A=self.inv_prob.A
        self.G=self.inv_prob.G
        self.data=self.inv_prob.data
        self.data_discrepancy=self.inv_prob.data_discrepancy
        self.n_latent=self.generative_model.n_latent
        self.noise_level=self.inv_prob.noise_level
        self.aim=self.inv_prob.aim


    # def gd_z(self, alpha=0.1,initial_z=None, line_search=False, step_size=0.01, iteration_number=1000 ):
    #     if initial_z is None:
    #         initial_z = np.random.normal(0,1,(1,self.n_latent))        
    #     f1= odl.solvers.L2NormSquared(self.y_space).translated(self.data)
    #     AG=odl.operator.operator.OperatorComp(self.A, self.G)
    #     f2=odl.solvers.FunctionalComp(f1,AG)
    #     f3=odl.solvers.L2NormSquared(self.z_space)
    #     f=f2+alpha*f3
        
    #     z=self.z_space.element(initial_z)
    #     callback=odl.solvers.util.callback.CallbackShowConvergence(f)
    #     linesearch=odl.solvers.util.steplen.BacktrackingLineSearch(f, max_num_iter=30)
    #     if line_search:
    #         try:
    #             odl.solvers.smooth.gradient.steepest_descent(f, z,line_search=linesearch, callback=callback, tol=1e-2, maxiter=iteration_number)
    #         except ValueError:
    #             print('number of iterations exceeded maximum without finding a sufficient decrease')
    #     odl.solvers.smooth.gradient.steepest_descent(f, z0,line_search=step_size, callback=callback, tol=1e-12, maxiter=iteration_number)
    
    #     self.G(z).show()
    #     return(z)
    
    def gd_backtracking_z(self, alpha=0.1, initial_z=None, iteration_number=100):
        if initial_z is None:
            initial_z = np.random.normal(0,1,(1,self.n_latent))
            
        
        opt_space=odl.space.pspace.ProductSpace( self.z_space,1)
        proj_z=odl.operator.pspace_ops.ComponentProjection(opt_space,0)
        
        
        f1= odl.solvers.L2NormSquared(self.y_space).translated(self.data)
        AG=odl.operator.operator.OperatorComp(self.A, odl.operator.operator.OperatorComp(self.G, proj_z))
        f2=odl.solvers.FunctionalComp(f1,AG)
        f3=odl.solvers.FunctionalComp(odl.solvers.L2NormSquared(self.z_space), proj_z)
        f=(1/(2*self.noise_level**2))*f2+alpha*f3
        g=[odl.solvers.ZeroFunctional(self.z_space)]
        
        final_result=opt_space.element(np.expand_dims(initial_z[0], axis=0))

        
        for i in range(initial_z.shape[0]):
            results=[]
            callback=odl.solvers.util.callback.CallbackStore(results=results)*(f)
            z_initial=opt_space.element(np.expand_dims(initial_z[i], axis=0))
            optimised=palm.PALM(f,g, callback=callback,niter=iteration_number, x=z_initial)
            if f(optimised.x)<f(final_result):
                final=optimised
                final_result=optimised.x
        print(final.counter)
        return(final.counter)
#        return( final.counter, self.G(proj_z(final.x)).asarray())


    
    
    
 
   
        
   
         # ||A(x)-y||_2^2+\alpha||z||^2_2+\beta||G(z)-x||_2^2
    def optim_x_soft_constraints(self,alpha, beta, initial_z=None, initial_x=None, iteration_number=100):
        if initial_z is None:
            initial_z = np.random.normal(0,1,(1,self.n_latent))
        if initial_x is None:
            initial_x = np.random.normal(0,1,(1,*self.generative_model.image_shape))  
        if len(initial_x.shape)==2:
            initial_x=np.expand_dims(initial_x, axis=0)
            
        zx_space=odl.space.pspace.ProductSpace(self.z_space, self.x_space)
        
        proj_z=odl.operator.pspace_ops.ComponentProjection(zx_space,0)
        proj_x=odl.operator.pspace_ops.ComponentProjection(zx_space,1)
        
        f1= odl.solvers.FunctionalComp(odl.solvers.FunctionalComp(odl.solvers.L2NormSquared(self.y_space).translated(self.data),self.A), proj_x)
        
        f2=alpha*odl.solvers.FunctionalComp(odl.solvers.L2NormSquared(self.z_space), proj_z)
        diffRange=odl.operator.operator.OperatorComp(self.G, proj_z)- proj_x
        f3=beta*odl.solvers.FunctionalComp(odl.solvers.L2NormSquared(self.x_space), diffRange)
        
        f=(1/(2*self.noise_level**2))*f1+f2+f3
        
        g1=odl.solvers.ZeroFunctional(self.z_space)
        g2=odl.solvers.ZeroFunctional(self.x_space)
        g=[g1,g2]
        
        final_result=zx_space.element((np.expand_dims(initial_z[0], axis=0), initial_x[0]))
     

        for i in range(initial_z.shape[0]):        
            results=[]
            callback=odl.solvers.util.callback.CallbackStore(results=results)*(f+g1*proj_z+g2*proj_x)
            zx_initial=zx_space.element((np.expand_dims(initial_z[i], axis=0), initial_x[i]))
            optimised=palm.PALM(f,g, callback=callback,niter=iteration_number, x=zx_initial)
            if f(optimised.x)+g1(proj_z(optimised.x))+g2(proj_x(optimised.x))<f(final_result)+g1(proj_z(final_result))+g2(proj_x(final_result)):
                final_result=optimised.x
                final=optimised
       # return(final.counter, proj_x(final.x).asarray())
        return(final.counter)
    

    
    def optim_x_tikhonov(self, beta, initial_x=None, iteration_number=100):
        if initial_x is None:
            initial_x = np.random.normal(0,1,(1,*self.generative_model.image_shape))
        if len(initial_x.shape)==2:
            initial_x=np.expand_dims(initial_x, axis=0)
        
        opt_space=odl.space.pspace.ProductSpace( self.x_space,1)
        proj_x=odl.operator.pspace_ops.ComponentProjection(opt_space,0)
        
        
        f1= odl.solvers.L2NormSquared(self.y_space).translated(self.data)
        f2=odl.solvers.FunctionalComp(f1,odl.operator.operator.OperatorComp(self.A, proj_x))
        f3=odl.solvers.FunctionalComp(odl.solvers.L2NormSquared(self.x_space), proj_x)
        f=(1/(2*self.noise_level**2))*f2+beta*f3
        g=[odl.solvers.ZeroFunctional(self.x_space)]
        
        final_result=opt_space.element(np.expand_dims(initial_x[0], axis=0))
        for i in range(initial_x.shape[0]):
            results=[]
            callback=odl.solvers.util.callback.CallbackStore(results=results)*(f)

            x_initial=opt_space.element(np.expand_dims(initial_x[i], axis=0))
            optimised=palm.PALM(f,g, callback=callback,niter=iteration_number, x=x_initial)
            
            if f(optimised.x)<f(final_result):
                final_result=optimised.x
                final=optimised
        return(final.counter)
       # return( final.counter, proj_x(final.x).asarray())

 


