# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 13:42:41 2020

@author: marga
"""
import numpy as np
import odl 

import odl.contrib.tensorflow
import odl.solvers
import palm 


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
        if self.noise_level<1e-6:
            self.noise_level=1
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
        
        
        print('check0')
        f1= odl.solvers.L2NormSquared(self.y_space).translated(self.data)
        print('check1')
        AG=odl.operator.operator.OperatorComp(self.A, odl.operator.operator.OperatorComp(self.G, proj_z))
        print('check2')
        f2=odl.solvers.FunctionalComp(f1,AG)
        print('check3')
        f3=odl.solvers.FunctionalComp(odl.solvers.L2NormSquared(self.z_space), proj_z)
        print('check4')
        f=(1/(2*self.noise_level**2))*f2+alpha*f3
        print('check5')
        g=[odl.solvers.ZeroFunctional(self.z_space)]
        print('check6')
        final_result=opt_space.element(np.expand_dims(initial_z[0], axis=0))

        
        for i in range(initial_z.shape[0]):
            results=[]
            callback=odl.solvers.util.callback.CallbackStore(results=results)*(f)
           
            z_initial=opt_space.element(np.expand_dims(initial_z[i], axis=0))
            optimised=palm.PALM(f,g, callback=callback,niter=iteration_number, x=z_initial)
            if f(optimised.x)<f(final_result):
                final_result=optimised.x

        return(proj_z(final_result))
    
    def gd_z_regularisation_parameter(self, alpha_min=0.1, alpha_max=1000, alpha_factor=0.5, initial_z=None, iteration_number=100, save_name='None', save=False, early_stop=False):
        if initial_z is None:
            initial_z = np.random.normal(0,1,(1,self.n_latent))
        alpha=alpha_max
        plot=np.zeros((1000,2))
        count=0
        if early_stop:
            stop=self.y_space.size*self.noise_level**2
        else:
            stop=0
  
        while alpha>alpha_min:
            z=self.gd_backtracking_z( alpha, initial_z, iteration_number)
            loss= np.linalg.norm(self.A(self.G(z))-self.data)**2

            if save:
 
                np.save(save_name+'lambda_'+str(alpha)+'.npy',self.G( z))   
                np.save(save_name+'_lambda_'+str(alpha)+'data.npy', self.A(self.G(z)))
            plot[count,:]=[alpha, loss]
            count+=1
            if loss< stop:                
                break
            else:
                alpha*=alpha_factor
        plot=plot[:count,:]
        if save:
            np.save(save_name+'_loss.npy', plot)
        return( plot)
    
    
    
    # ||A(G(z)+u)-y||_2^2+\lambda\mu||z||^2_2+\lamba||u||_1
    def optim_z_sparse_deviations(self,alpha=0.1, beta=0.8, initial_z=None, initial_u=None, iteration_number=50 ):
        if initial_z is None:
            initial_z = np.random.normal(0,1,(1,self.n_latent))
        if initial_u is None:
            initial_u = np.zeros((1, *self.generative_model.image_shape))
        if len(initial_u.shape)==2:
            initial_u=np.expand_dims(initial_u, axis=0)
        
        
        
        zu_space=odl.space.pspace.ProductSpace(self.z_space, self.x_space)
        
        proj_z=odl.operator.pspace_ops.ComponentProjection(zu_space,0)
        proj_x=odl.operator.pspace_ops.ComponentProjection(zu_space,1)
        g1= alpha*odl.solvers.L2NormSquared(self.z_space)
        g2=beta*odl.solvers.L1Norm(self.x_space)
        g=[g1,g2]
        
        recon= proj_x+odl.operator.operator.OperatorComp(self.G, proj_z)
        AGu=odl.operator.operator.OperatorComp(self.A, recon)
        f1= odl.solvers.L2NormSquared(self.y_space).translated(self.data)
        f=(1/(2*self.noise_level**2))*odl.solvers.FunctionalComp(f1,AGu)
        
        final_result=zu_space.element((np.expand_dims(initial_z[0], axis=0), initial_u[0]))
        

        for i in range(initial_z.shape[0]):
            results=[]
            callback=odl.solvers.util.callback.CallbackStore(results=results)*(f+g1*proj_z+g2*proj_x)
            zu_initial=zu_space.element((np.expand_dims(initial_z[i], axis=0), initial_u[i]))
            optimised=palm.PALM(f,g, callback=callback,niter=iteration_number, x=zu_initial)
            if f(optimised.x)+g1(proj_z(optimised.x))+g2(proj_x(optimised.x))<f(final_result)+g1(proj_z(final_result))+g2(proj_x(final_result)):
                final_result=optimised.x
        return(proj_z(final_result), proj_x(final_result))
   
         # ||A(G(z)+u)-y||_2^2+\lambda\mu||z||^2_2+\lamba||u||_1
    def optim_z_sparse_regularisation_parameter(self, lambda_min=0.1, lambda_max=1000, lambda_factor=0.5, mu_min=0.01, mu_max=10, mu_factor=0.5,initial_z=None, initial_u=None, iteration_number=100, save_name='None', early_stop=True, save=False):
        if initial_z is None:
            initial_z = np.random.normal(0,1,(1,self.n_latent))
        if initial_u is None:
            initial_u = np.zeros((1, *self.generative_model.image_shape))
        if len(initial_u.shape)==2:
            initial_u=np.expand_dims(initial_u, axis=0)
        if early_stop:
            stop=self.y_space.size*self.noise_level**2
        else:
            stop=0
        lamb=lambda_max
        mu=mu_max
        plot=np.zeros((1000,3))
        count=0
        inner=False
        while mu>mu_min:

 
            inner_count=0
            while lamb>lambda_min:
                z,u=self.optim_z_sparse_deviations( lamb*mu,lamb, initial_z, initial_u, iteration_number)
                loss=np.linalg.norm(self.A(self.G(z)+u)-self.data)**2

                if save:

                    np.save(save_name+'lambda_'+str(lamb)+'_mu_'+str(mu)+'.npy',self.G( z)+u)
                    np.save(save_name+'_lambda_'+str(lamb)+'_mu_'+str(mu)+'data.npy', self.A(self.G(z)+u))
                plot[count,:]=[lamb, mu, loss]
                count+=1
                inner_count+=1
                if loss< stop:
                    inner=True
                    break
                else:
                    lamb*=lambda_factor
 
            if inner:
                print('Found a value of lambda')

                break
            else:
                mu*=mu_factor
                lamb=lambda_max
                print('Trying another value of mu')

                   
        plot=plot[:count,:]      

        if save:
            np.save(save_name+'lambda_mu_loss.npy', plot)
        return(plot)
        
   
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
        
        f1= (1/(2*self.noise_level**2))*odl.solvers.FunctionalComp(odl.solvers.FunctionalComp(odl.solvers.L2NormSquared(self.y_space).translated(self.data),self.A), proj_x)
        
        f2=alpha*odl.solvers.FunctionalComp(odl.solvers.L2NormSquared(self.z_space), proj_z)
        diffRange=odl.operator.operator.OperatorComp(self.G, proj_z)- proj_x
        f3=beta*odl.solvers.FunctionalComp(odl.solvers.L2NormSquared(self.x_space), diffRange)
        
        f=f1+f2+f3
        
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
            
        return(proj_z(final_result),proj_x(final_result))
    
                # ||A(x)-y||_2^2+\lambda*\mu||z||^2_2+\lambda||G(z)-x||_2^2
    def optim_x_soft_constraints_regularisation_parameter(self, lambda_min=0.1, lambda_max=1000, lambda_factor=0.5, mu_min=0.01, mu_max=10, mu_factor=0.5,initial_z=None, initial_x=None, iteration_number=100, save_name='None', early_stop=True, save=False):
        if initial_z is None:
            initial_z = np.random.normal(0,1,(1,self.n_latent))
        if initial_x is None:
            initial_x = np.random.normal(0,1,(1,*self.generative_model.image_shape))  
        if len(initial_x.shape)==2:
            initial_x=np.expand_dims(initial_x, axis=0)
        if early_stop:
            stop=self.y_space.size*self.noise_level**2
        else:
            stop=0
        lamb=lambda_max
        mu=mu_max
        plot=np.zeros((1000,3))
        count=0
        inner=False
        while mu>mu_min:

            inner_count=0
            while lamb>lambda_min:
                z,x=self.optim_x_soft_constraints( lamb*mu,lamb, initial_z, initial_x, iteration_number)
                loss=np.linalg.norm(self.A(x)-self.data)**2

                if save:

                    np.save(save_name+'lambda_'+str(lamb)+'_mu_'+str(mu)+'.npy', x)
                    np.save(save_name+'_lambda_'+str(lamb)+'_mu_'+str(mu)+'data.npy', self.A(x))

                plot[count,:]=[lamb, mu, loss]
                count+=1
                inner_count+=1

                if loss< stop:
                    inner=True
                    break
                else:
                    lamb*=lambda_factor

            if inner:
                break
               
            else:
                mu*=mu_factor
                lamb=lambda_max
        plot=plot[:count,:]
        if save:
            np.save(save_name+'lambda_mu_loss.npy', plot)
        
        return(plot)
    
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

        return(proj_x(final_result))
    
    def optim_x_tik_regularisation_parameter(self, beta_min=0.1, beta_max=1000, beta_factor=0.5, initial_x=None, iteration_number=100, save_name='None', save=False, early_stop=True):
        if initial_x is None:
            initial_x = np.random.normal(0,1,(1,*self.generative_model.image_shape))
        if len(initial_x.shape)==2:
            initial_x=np.expand_dims(initial_x, axis=0)
        if early_stop:
            stop=self.y_space.size*self.noise_level**2
        else:
            stop=0
        beta=beta_max
        plot=np.zeros((1000,2))
        count=0
        while beta>beta_min:
            x=self.optim_x_tikhonov( beta, initial_x, iteration_number)
            loss=np.linalg.norm(self.A(x)-self.data)**2
            plot[count,:]=[beta, loss]
            if save:

                np.save(save_name+'_lambda_'+str(beta)+'.npy', x)
                np.save(save_name+'_lambda_'+str(beta)+'data.npy', self.A(x))
            count+=1
            if loss< stop:
                break
            else:
                beta*=beta_factor
        plot=plot[:count,:]        
        if save:
            np.save(save_name+'loss.npy', plot)
        return(plot)
