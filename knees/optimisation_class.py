# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 13:42:41 2020

@author: marga
"""
import numpy as np
import tensorflow as tf
import odl 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import odl.contrib.tensorflow
import odl.solvers
import palm 


SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

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
    #     odl.solvers.smooth.gradient.steepest_descent(f, z,line_search=step_size, callback=callback, tol=1e-12, maxiter=iteration_number)
    
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
        fig = plt.figure(figsize=(16, 16))
        fig2=plt.figure(figsize=(16, 16))
        gs = gridspec.GridSpec(4, 4)
        
        
        for i in range(initial_z.shape[0]):
            results=[]
            callback=odl.solvers.util.callback.CallbackStore(results=results)*(f)
           
            z_initial=opt_space.element(np.expand_dims(initial_z[i], axis=0))
            optimised=palm.PALM(f,g, callback=callback,niter=iteration_number, x=z_initial)
            if f(optimised.x)<f(final_result):
                final_result=optimised.x
            ax = fig.add_subplot(gs[i])
            im=ax.imshow(self.G( proj_z(optimised.x)),  cmap='gray')
            ax.set_title('Loss = {0:2.5f}'.format(f(optimised.x)))
            fig.colorbar(im, ax=ax)
            try:
                ax2 = fig2.add_subplot(gs[i], sharey=ax2)
            except:
                ax2=fig2.add_subplot(gs[i])
            ax2.scatter(range(5,len(results)),results[5:])
            ax2.set_title('Convergence')
        self.G( proj_z(final_result)).show()
        return(proj_z(final_result))
    
    def gd_z_regularisation_parameter(self, alpha_min=0.00001, alpha_max=10, alpha_factor=0.5, initial_z=None, iteration_number=100, save_name='None', save=False):
        if initial_z is None:
            initial_z = np.random.normal(0,1,(1,self.n_latent))
        alpha=alpha_max
        plot=np.zeros((1000,2))
        count=0
        fig = plt.figure(figsize=(20, 16))
        fig2 = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(5, 4)    
        while alpha>alpha_min:
            z=self.gd_backtracking_z( alpha, initial_z, iteration_number)
            loss= np.linalg.norm(self.A(self.G(z))-self.data)**2
            ax = fig.add_subplot(gs[count])
            im=ax.imshow(self.G( z),  cmap='gray')
            ax.set_title('Lambda={0:2.5f}, Loss = {1:2.5f}'.format(alpha,loss))
            fig.colorbar(im, ax=ax)
            if save:
                fig3=plt.figure(figsize=(7,7))
                im=plt.imshow(self.G( z),  cmap='gray')
                fig3.colorbar(im)
                plt.title('$lambda$={0:2.5f}'.format(alpha))
                plt.xlabel('Loss = {0:2.5f}'.format(loss))
                fig3.savefig(save_name+'lambda_'+str(alpha)+'.png')         
            ax = fig2.add_subplot(gs[count])
            im=ax.imshow(self.G( z)-self.aim,  cmap='gray')
            ax.set_title('Lambda={0:2.5f}, Loss = {1:2.5f}'.format(alpha,loss))
            fig2.colorbar(im, ax=ax)
            plot[count,:]=[alpha, loss]
            count+=1
            if loss< self.y_space.size*self.noise_level**2:                
                break
            else:
                alpha*=alpha_factor
        plot=plot[:count,:]
        if save:
            fig.savefig(save_name+'.png')
            fig2.savefig(save_name+'_loss.png')
        fig3=plt.figure()
        plt.scatter(plot[:,0], plot[:,1])
        plt.hlines(self.y_space.size*self.noise_level**2, alpha_min, alpha_max)
        plt.xscale('log')
        plt.xlabel('lambda')
        plt.ylabel('Data discrepancy')
        if save:   
            plt.savefig(save_name+'_loss.png')
        return(fig, fig2, plot)
        
            
        
    # ||A(G(z)+u)-y||_2^2+$lambda$$\mu$||z||^2_2+\lamba||u||_1
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
        
        fig = plt.figure(figsize=(16, 16))
        fig2=plt.figure(figsize=(16, 16))
        gs = gridspec.GridSpec(4, 4)
        for i in range(initial_z.shape[0]):
            results=[]
            callback=odl.solvers.util.callback.CallbackStore(results=results)*(f+g1*proj_z+g2*proj_x)
            zu_initial=zu_space.element((np.expand_dims(initial_z[i], axis=0), initial_u[i]))
            optimised=palm.PALM(f,g, callback=callback,niter=iteration_number, x=zu_initial)
            if f(optimised.x)+g1(proj_z(optimised.x))+g2(proj_x(optimised.x))<f(final_result)+g1(proj_z(final_result))+g2(proj_x(final_result)):
                final_result=optimised.x
            ax = fig.add_subplot(gs[i])
            im=ax.imshow(recon(optimised.x),  cmap='gray')
            ax.set_title('Loss = {0:2.5f}'.format(f(optimised.x)+g1(proj_z(optimised.x))+g2(proj_x(optimised.x))))
            fig.colorbar(im, ax=ax)
            try:
                ax2 = fig2.add_subplot(gs[i], sharey=ax2)
            except:
                ax2=fig2.add_subplot(gs[i])
            ax2.scatter(range(5,len(results)),results[5:])
            ax2.set_title('Convergence')
            
            
        recon(final_result).show()
        proj_x(final_result).show('Sparse Addition')
        self.G(proj_z(final_result)).show('Result from generator')
        return(proj_z(optimised.x), proj_x(optimised.x))
   
     # ||A(G(z)+u)-y||_2^2+$lambda$$\mu$||z||^2_2+\lamba||u||_1
    def optim_z_sparse_regularisation_parameter(self, lambda_min=0.00001, lambda_max=10, lambda_factor=0.5, mu_min=0.01, mu_max=2, mu_factor=0.5,initial_z=None, initial_u=None, iteration_number=100, save_name='None', early_stop=True, save=False):
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
            fig = plt.figure(figsize=(20, 16))
            gs = gridspec.GridSpec(5, 4)
            fig2 = plt.figure(figsize=(20, 16))
 
            inner_count=0
            while lamb>lambda_min:
                z,u=self.optim_z_sparse_deviations( lamb*mu,lamb, initial_z, initial_u, iteration_number)
                loss=np.linalg.norm(self.A(self.G(z)+u)-self.data)**2
                ax = fig.add_subplot(gs[inner_count])
                im=ax.imshow(self.G( z)+u,  cmap='gray')
                ax.set_title('Lambda={0:2.5f},mu={1:2.5f}, Loss = {2:2.5f}'.format(lamb,mu,loss))
                fig.colorbar(im, ax=ax)
                if save:
                    fig3=plt.figure(figsize=(7,7))
                    im=plt.imshow(self.G( z)+u,  cmap='gray')
                    fig3.colorbar(im)
                    plt.title('$lambda$={0:2.5f},$\mu$={1:2.5f}'.format(lamb, mu))
                    plt.xlabel('Loss = {0:2.5f}'.format(loss))
                    fig3.savefig(save_name+'lambda_'+str(lamb)+'_mu_'+str(mu)+'.png')
                ax = fig2.add_subplot(gs[inner_count])
                im=ax.imshow(self.G( z)+u-self.data,  cmap='gray')
                ax.set_title('Lambda={0:2.5f},mu={1:2.5f}, Loss = {2:2.5f}'.format(lamb,mu,loss)) 
                fig2.colorbar(im, ax=ax)
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
                if save:
                    fig.savefig(save_name+'_'+str(mu)+'.png')
                    fig2.savefig(save_name+'_'+str(mu)+'Loss.png')
                break
            else:
                mu*=mu_factor
                lamb=lambda_max
                print('Trying another value of mu')
                if save:
                    fig.savefig(save_name+'_'+str(mu)+'.png')
                    fig2.savefig(save_name+'_'+str(mu)+'Loss.png')
                
        plot=plot[:count,:]      
        plt.figure()
        plt.scatter(plot[:,0], plot[:,2], c=plot[:,1])
        plt.colorbar()
        plt.hlines(self.y_space.size*self.noise_level**2, lambda_min, lambda_max)
        plt.xscale('log')
        plt.xlabel('lambda')
        plt.ylabel('Data discrepancy')
        if save:
            plt.savefig(save_name+'lambda_loss.png')
        plt.figure()
        plt.scatter(plot[:,1], plot[:,2])
        plt.hlines(self.y_space.size*self.noise_level**2, mu_min, mu_max)
        plt.xscale('log')
        plt.xlabel('mu')
        
        plt.ylabel('Data discrepancy')
        if save:
            plt.savefig(save_name+'mu_loss.png')
        return(plot,fig, fig2)
        
   
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
     
        fig = plt.figure(figsize=(16, 16))
        fig2=plt.figure(figsize=(16, 16))
        gs = gridspec.GridSpec(4, 4)
        for i in range(initial_z.shape[0]):        
            results=[]
            callback=odl.solvers.util.callback.CallbackStore(results=results)*(f+g1*proj_z+g2*proj_x)
            zx_initial=zx_space.element((np.expand_dims(initial_z[i], axis=0), initial_x[i]))
            optimised=palm.PALM(f,g, callback=callback,niter=iteration_number, x=zx_initial)
            if f(optimised.x)+g1(proj_z(optimised.x))+g2(proj_x(optimised.x))<f(final_result)+g1(proj_z(final_result))+g2(proj_x(final_result)):
                final_result=optimised.x
            ax = fig.add_subplot(gs[i])
            im=ax.imshow(proj_x(optimised.x),  cmap='gray')
            ax.set_title('Loss = {0:2.5f}'.format(f(optimised.x)+g1(proj_z(optimised.x))+g2(proj_x(optimised.x))))
            fig.colorbar(im, ax=ax)
            try:
                ax2 = fig2.add_subplot(gs[i], sharey=ax2)
            except:
                ax2=fig2.add_subplot(gs[i])
            ax2.scatter(range(5,len(results)),results[5:])
            ax2.set_title('Convergence')
        
        proj_x(final_result).show()
        return(proj_z(final_result),proj_x(final_result))
    
                # ||A(x)-y||_2^2+$lambda$*$\mu$||z||^2_2+$lambda$||G(z)-x||_2^2
    def optim_x_soft_constraints_regularisation_parameter(self, lambda_min=0.0001, lambda_max=2, lambda_factor=0.5, mu_min=0.01, mu_max=2, mu_factor=0.5,initial_z=None, initial_x=None, iteration_number=100, save_name='None', early_stop=True, save=False):
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
            fig = plt.figure(figsize=(20, 16))
            fig2 = plt.figure(figsize=(20, 16))
            gs = gridspec.GridSpec(5, 4)
            inner_count=0
            while lamb>lambda_min:
                z,x=self.optim_x_soft_constraints( lamb*mu,lamb, initial_z, initial_x, iteration_number)
                loss=np.linalg.norm(self.A(x)-self.data)**2
                ax = fig.add_subplot(gs[inner_count])
                im=ax.imshow(x,  cmap='gray')
                ax.set_title('Lambda={0:2.5f},mu={1:2.5f}, Loss = {2:2.5f}'.format(lamb,mu, loss))
                fig.colorbar(im, ax=ax)
                if save:
                    fig3=plt.figure(figsize=(7,7))
                    im=plt.imshow(x,  cmap='gray')
                    fig3.colorbar(im)
                    plt.title('$lambda$={0:2.5f},$\mu$={1:2.5f}'.format(lamb, mu))
                    plt.xlabel('Loss = {0:2.5f}'.format(loss))
                    fig3.savefig(save_name+'lambda_'+str(lamb)+'_mu_'+str(mu)+'.png')
                ax = fig2.add_subplot(gs[inner_count])
                im=ax.imshow(x-self.data,  cmap='gray')
                ax.set_title('Lambda={0:2.5f},mu={1:2.5f}, Loss = {2:2.5f}'.format(lamb,mu, loss))
                fig2.colorbar(im, ax=ax)
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
                if save:
                    fig.savefig(save_name+str(lamb)+'_'+str(mu)+'.png')
                    fig2.savefig(save_name+str(lamb)+'_'+str(mu)+'Loss.png')
            else:
                mu*=mu_factor
                lamb=lambda_max
                if save:
                    fig.savefig(save_name+str(lamb)+'_'+str(mu)+'.png')
                    fig2.savefig(save_name+str(lamb)+'_'+str(mu)+'Loss.png')
                
        plot=plot[:count,:]        
        plt.figure()
        plt.scatter(plot[:,0], plot[:,2],c=plot[:,1])
        plt.colorbar()
        plt.hlines(self.y_space.size*self.noise_level**2, lambda_min, lambda_max)
        plt.xscale('log')
        plt.xlabel('lambda')
        plt.ylabel('Data discrepancy')
        if save:
            plt.savefig(save_name+'lambda_loss.png')
        plt.figure()
        plt.scatter(plot[:,1], plot[:,2])
        plt.hlines(self.y_space.size*self.noise_level**2, mu_min, mu_max)
        plt.xscale('log')
        plt.xlabel('mu')
        plt.ylabel('Data discrepancy')
        if save:
            plt.savefig(save_name+'mu_loss.png')
        return(plot, fig, fig2)
    
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
        fig = plt.figure(figsize=(16, 16))
        fig2=plt.figure(figsize=(16, 16))
        gs = gridspec.GridSpec(4, 4)
        
        
        for i in range(initial_x.shape[0]):
            results=[]
            callback=odl.solvers.util.callback.CallbackStore(results=results)*(f)
           
            x_initial=opt_space.element(np.expand_dims(initial_x[i], axis=0))
            optimised=palm.PALM(f,g, callback=callback,niter=iteration_number, x=x_initial)
            if f(optimised.x)<f(final_result):
                final_result=optimised.x
            ax = fig.add_subplot(gs[i])
            im=ax.imshow( proj_x(optimised.x),  cmap='gray')
            ax.set_title('Loss = {0:2.5f}'.format(f(optimised.x)))
            fig.colorbar(im, ax=ax)
            try:
                ax2 = fig2.add_subplot(gs[i], sharey=ax2)
            except:
                ax2=fig2.add_subplot(gs[i])
            ax2.scatter(range(5,len(results)),results[5:])
            ax2.set_title('Convergence')
        proj_x(final_result).show()
        return(proj_x(final_result))
    
    def optim_x_tik_regularisation_parameter(self, beta_min=0.00001, beta_max=10, beta_factor=0.5, initial_x=None, iteration_number=100, save_name='None', save=False, early_stop=True):
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
        fig = plt.figure(figsize=(20, 16))
        fig2 = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(5, 4)
        while beta>beta_min:
            x=self.optim_x_tikhonov( beta, initial_x, iteration_number)
            loss=np.linalg.norm(self.G(self.A(x))-self.data)**2
            ax = fig.add_subplot(gs[count])
            im=ax.imshow(x,  cmap='gray')
            ax.set_title('Lambda={0:2.5f}, Loss = {1:2.5f}'.format(beta,loss))
            fig.colorbar(im, ax=ax)
            if save:
                fig3=plt.figure(figsize=(7,7))
                im=plt.imshow(x,  cmap='gray')
                fig3.colorbar(im)
                plt.title('$lambda$={0:2.5f}'.format( beta))
                plt.xlabel('Loss = {0:2.5f}'.format(loss))
                fig3.savefig(save_name+'_lambda_'+str(beta)+'.png')
            ax = fig2.add_subplot(gs[count])
            im=ax.imshow(x-self.aim,  cmap='gray')
            ax.set_title('Lambda={0:2.5f}, Loss = {1:2.5f}'.format(beta,loss))
            fig2.colorbar(im, ax=ax)
            plot[count,:]=[beta, loss]
            count+=1
            if loss< stop:
                
                break
            else:
                beta*=beta_factor
        plot=plot[:count,:]        
        fig.savefig(save_name+'.png')
        fig2.savefig(save_name+'loss.png')
        plt.figure()
        plt.scatter(plot[:,0], plot[:,1])
        plt.hlines(self.y_space.size*self.noise_level**2, beta_min, beta_max)
        plt.xscale('log')
        plt.xlabel('beta')
        plt.ylabel('Data discrepancy')
        if save:
            plt.savefig(save_name+'loss.png')
        return(plot, fig, fig2)
    # def optim_x_tv(self, beta, initial_x=None, iteration_number=100):
    #     if initial_x is None:
    #         initial_x = np.random.normal(0,1,(1,*self.generative_model.image_shape))
    #     if len(initial_x.shape)==2:
    #         initial_x=np.expand_dims(initial_x, axis=0)
        
    #     opt_space=odl.space.pspace.ProductSpace( self.x_space,1)
    #     proj_x=odl.operator.pspace_ops.ComponentProjection(opt_space,0)
        
        
    #     f1= odl.solvers.L2NormSquared(self.y_space).translated(self.data)
    #     f2=odl.solvers.FunctionalComp(f1,odl.operator.operator.OperatorComp(self.A, proj_x))
    #     f=f2
        
    #     grad=odl.Gradient(self.x_space)
        
    #     g=[beta*odl.solvers.FunctionalComp( odl.solvers.GroupL1Norm(grad.range), grad)]
        
    #     final_result=opt_space.element(np.expand_dims(initial_x[0], axis=0))
    #     fig = plt.figure(figsize=(16, 16))
    #     fig2=plt.figure(figsize=(16, 16))
    #     gs = gridspec.GridSpec(4, 4)
        
        
    #     for i in range(initial_x.shape[0]):
    #         results=[]
    #         callback=odl.solvers.util.callback.CallbackStore(results=results)*(f)
           
    #         x_initial=opt_space.element(np.expand_dims(initial_x[i], axis=0))
    #         optimised=palm.PALM(f,g, callback=callback,niter=iteration_number, x=x_initial)
    #         if f(optimised.x)<f(final_result):
    #             final_result=optimised.x
    #         ax = fig.add_subplot(gs[i])
    #         im=ax.imshow( proj_x(optimised.x),  cmap='gray')
    #         ax.set_title('Loss = {0:2.5f}'.format(f(optimised.x)))
    #         fig.colorbar(im, ax=ax)
    #         try:
    #             ax = fig2.add_subplot(gs[i], sharey=ax)
    #         except:
    #             ax=fig2.add_subplot(gs[i])
    #         ax.scatter(range(5,len(results)),results[5:])
    #         ax.set_title('Convergence')
    #     proj_x(final_result).show()
    #     return(proj_x(final_result))   