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
import cv2

class optimisation():
    def __init__(self,inv_prob, generative_model,sess):
        self.inv_prob=inv_prob
        self.generative_model=generative_model
        self.session=sess
    def plot_results(self, z, save=False, save_name='None'):
        samples=self.generative_model.generate(z)
        fig = plt.figure(figsize=(16, 16))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.4, hspace=0.4)
        for i in range(np.shape(samples)[0]):
            ax = plt.subplot(gs[i])
            plt.imshow(samples[i], vmin=0, vmax=1)
            ax.title.set_text('z1=%0.3f, z2=%0.3f, Loss=%0.3f' %(z[i,0], z[i,1], self.inv_prob.data_term(samples[i]).eval()))
            plt.axis('off')
        if save:
            plt.savefig(save_name+'.jpg')
        #plt.colorbar()
        plt.show()
    def plot_results_plus_sparse(self, z1, u1, save=False, save_name='None'):
    
        samples=self.generative_model.generate(z1)+u1
        fig = plt.figure(figsize=(16, 16))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.4, hspace=0.4)
        for i in range(np.shape(z1)[0]):
            ax = plt.subplot(gs[i])
            plt.imshow(samples[i], vmin=0, vmax=1)
            ax.title.set_text('z1=%0.3f, z2=%0.3f, Loss=%0.3f' %(z1[i,0], z1[i,1], self.inv_prob.data_term(samples[i]).eval()))
        if save:
            plt.savefig(save_name+'.jpg')
        #plt.colorbar()
        plt.show()
        
    def optim_z(self,alpha, initial_z, iteration_number,  initial_L=20, upper=5, lower=0.6, plot=False, early_stop=False, save_name='None' ):
        objective_tensor=self.inv_prob.data_term(self.generative_model.reconstruction)+alpha*tf.nn.l2_loss(self.generative_model.z)
        grad=tf.gradients(objective_tensor, self.generative_model.z)[0]
        def objective(z):
            return(self.session.run(objective_tensor, feed_dict={self.generative_model.z:z, self.generative_model.keep_prob:1.0}))
        
        def gradient(z):
            return(self.session.run(grad, feed_dict={self.generative_model.z:z, self.generative_model.keep_prob:1.0}))
        
        z_0=initial_z
        number=np.shape(initial_z)[0]
        L=initial_L
        #Keep hold of loss values to plot later 
        loss_plot=np.zeros(iteration_number+1)
        loss_plot[0]=self.inv_prob.data_term(self.generative_model.generate(z_0)).eval()/number
        #print and plot the initial set up 
        print('iter %4d, l2_loss=%0.3f'  %(0, loss_plot[0]))
        stop=np.ones(np.shape(initial_z)[0])
                   
        if plot:
                    self.plot_results(z_0, save=False, save_name=save_name+'initial')
        count=0
        for i in range(100000):
            z_1=np.zeros(np.shape(z_0))
            z_1[stop==1]=z_0[stop==1]-(1/L)*gradient(z_0[stop==1])
            if objective(z_1[stop==1])<=objective(z_0[stop==1])         +(1/(2*100*L))*np.linalg.norm(gradient(z_0), ord=2)**2:
                    z_0[stop==1]=z_1[stop==1]
                    count+=1
                    L=lower*L
                    loss_plot[count]=self.inv_prob.data_term(self.generative_model.generate(z_0)).eval()/number
            else:
                L=upper*L
        #           print(z_1-z_0)
         
            if (count+1)%30==0:
                if plot:
                    self.plot_results(z_0)
                print('iter %4d, l2_loss=%0.3f' %(count+1, loss_plot[count]))  
                
            if early_stop:
                for i in range(np.shape(initial_z)[0]):
                    if objective(z_0[i,:])<=early_stop:
                        stop[i]=0
            if count==iteration_number:
                    break
            if np.sum(stop)==0:
                loss_plot=loss_plot[:count+1]
                if plot:
                    self.plot_results(z_0)
                break 
            if np.abs(np.mean(gradient(z_0[stop==1])))<10**(-4):
                print('Small gradient, count =',count)
                print(np.mean(gradient(z_0[stop==1])))
                loss_plot=loss_plot[:count+1]
                if plot:
                    self.plot_results(z_0)
                break 
        print('After optimisation: latent variable is ', z_0)
        
        if plot:
                self.plot_results(z_0, save=False, save_name=str(save_name+'backtracking'))
                plt.plot(range(count+1), loss_plot)
                plt.xlabel('Iteration Number')
                plt.ylabel('Loss')
                plt.title(save_name)
                plt.savefig(save_name+'loss_plt.jpg')
            
        return z_0, loss_plot
    
    def optim_z_odlgd(self, alpha, initial_z, iteration_number=1000, plot=False):
        z=u=tf.placeholder(dtype=tf.float32,shape=np.shape(initial_z), name='z_opt')
        objective_tensor_z=self.inv_prob.data_term(self.generative_model.generator(z))+alpha*tf.nn.l2_loss(z)
        grad_z=tf.gradients(objective_tensor_z, z)[0]
        odl_image_loss=odl.contrib.tensorflow.TensorflowOperator(z, objective_tensor_z, sess=self.session)
        odl_image_loss.gradient=odl.contrib.tensorflow.TensorflowOperator(z, grad_z, sess=self.session)
        space=odl.rn(initial_z.shape,dtype='float32')
        z0=space.element(initial_z)
        results = []
            #callback = odl.solvers.util.callback.CallbackStore(results=results)
            #callback=odl.solvers.util.callback.CallbackPrintNorm()
        line_search =odl.solvers.util.steplen.BacktrackingLineSearch(odl_image_loss)
        callback=odl.solvers.util.callback.CallbackShowConvergence(odl_image_loss)
        odl.solvers.smooth.gradient.steepest_descent(f=odl_image_loss,x=z0, line_search=1.0, maxiter=iteration_number, callback=callback)
        if plot:
            self.plot_results(z_0, save=False,  save_name=str(save_name+'gd'))

                
        return(z0, results)
    
    # ||A(G(z)+u)||_2^2+\alpha||z||^2_2+\beta||u||_1
    def optim_z_sparse_deviations(self,alpha, beta, initial_z, initial_u, iteration_number, alternations=3,  lower=0.6, plot=False,  save_name='None'):
        z=u=tf.placeholder(dtype=tf.float32,shape=np.shape(initial_z), name='z_opt')
        u=tf.placeholder(dtype=tf.float32,shape=np.shape(initial_u), name='u_opt')
              
        
        def pgd(objective, gradient, input_tensor, initial_tensor,  reg_param=beta, gamma=0.01, niter=100):
            odl_image_loss=odl.contrib.tensorflow.TensorflowOperator(input_tensor, objective, sess=self.session)
            odl_image_loss.gradient=odl.contrib.tensorflow.TensorflowOperator(input_tensor, gradient, sess=self.session)
            space=odl.rn(initial_tensor.shape,dtype='float32')
            l1_norm = reg_param*odl.solvers.L1Norm(space)
            x=space.element(initial_tensor)
            results = []
            #callback = odl.solvers.util.callback.CallbackStore(results=results)
            #callback=odl.solvers.util.callback.CallbackPrintNorm()
            callback=odl.solvers.util.callback.CallbackShowConvergence(odl_image_loss)
            odl.solvers.proximal_gradient(x, f=l1_norm, g=odl_image_loss, niter=niter, gamma=gamma, callback=callback)
            return(x, results)

        z_opt=initial_z
        u_opt=initial_u
        results=[]
        for i in range(3):
            objective_tensor_z=self.inv_prob.data_term(self.generative_model.generator(z)+u_opt)+alpha*tf.nn.l2_loss(z)
            input_tensor_z=z
            grad_z=tf.gradients(objective_tensor_z, z)[0]
            z_opt,results_1=pgd(objective_tensor_z, grad_z, input_tensor_z, initial_tensor=z_opt,  reg_param=0, gamma=0.01, niter=100)
            #results=np.concatenate(results, results_1)
            self.plot_results_plus_sparse(z_opt, u_opt)
            
            objective_tensor_u=self.inv_prob.data_term(self.generative_model.generate(z_opt)+u)+alpha*tf.nn.l2_loss(z_opt)
            input_tensor_u=u
            grad_u=tf.gradients(objective_tensor_u, u)[0]
            u_opt,results_1=pgd(objective_tensor_u, grad_u, input_tensor=u, initial_tensor=u_opt,  reg_param=beta, gamma=0.01, niter=100)
            #results=np.concatenate(results, results_1)
            self.plot_results_plus_sparse(z_opt, u_opt)
        return(z_opt, u_opt)
    
        # ||A(G(z)+u)-y||_2^2+\alpha||z||^2_2+\beta||u||_1
    def palm_z_u(self,alpha, beta, initial_z, initial_u, iteration_number, plot=False, gamma=10 ,  save_name='None' ):
        z=u=tf.placeholder(dtype=tf.float32,shape=np.shape(initial_z), name='z_opt')
        u=tf.placeholder(dtype=tf.float32,shape=np.shape(initial_u), name='u_opt')
        objective_tensor=self.inv_prob.data_term(self.generative_model.generator(z)+u)
        grad_z=tf.gradients(objective_tensor, z)[0]
        grad_u=tf.gradients(objective_tensor, u)[0]
        
        space_z=odl.rn(initial_z.shape,dtype='float32')
        space_u=odl.rn(initial_u.shape,dtype='float32')
        prox_u=odl.solvers.nonsmooth.proximal_operators.proximal_l1(space_u, lam=beta)
        update_u=prox_u(gamma)
        prox_z=odl.solvers.nonsmooth.proximal_operators.proximal_l2_squared(space_z, lam=alpha)
        update_z=prox_z(gamma)
        
        z_opt=initial_z
        u_opt=initial_u
        results_hold=np.zeros(iteration_number)
        for i in range(iteration_number):
            z_opt=update_z(z_opt-(1/gamma)*grad_z.eval(session=self.session, feed_dict={z:z_opt, u:u_opt}))
            u_opt=update_u(u_opt-(1/gamma)*grad_u.eval(session=self.session, feed_dict={z:z_opt, u:u_opt}))  
            results_hold[i]=objective_tensor.eval(session=self.session, feed_dict={z:z_opt, u:u_opt})
            if i%10==0:
                if plot:
                    self.plot_results_plus_sparse(z_opt, u_opt)
        plt.plot(range(iteration_number), results_hold)
        return(u_opt, z_opt)
            
            # ||A(x)-y||_2^2+\alpha||z||^2_2+\beta||G(z)-x||_1
    def palm_z_x(self,alpha, beta, initial_z, initial_u, iteration_number, plot=False, gamma=10 ,  save_name='None' ):
        z=u=tf.placeholder(dtype=tf.float32,shape=np.shape(initial_z), name='z_opt')
        u=tf.placeholder(dtype=tf.float32,shape=np.shape(initial_u), name='u_opt')
        objective_tensor=self.inv_prob.data_term(self.generative_model.generator(z)+u)
        grad_z=tf.gradients(objective_tensor, z)[0]
        grad_u=tf.gradients(objective_tensor, u)[0]
        
        space_z=odl.rn(initial_z.shape,dtype='float32')
        space_u=odl.rn(initial_u.shape,dtype='float32')
        prox_u=odl.solvers.nonsmooth.proximal_operators.proximal_l1(space_u, lam=beta)
        update_u=prox_u(gamma)
        prox_z=odl.solvers.nonsmooth.proximal_operators.proximal_l2_squared(space_z, lam=alpha)
        update_z=prox_z(gamma)
        
        z_opt=initial_z
        u_opt=initial_u
        results_hold=np.zeros(iteration_number)
        for i in range(iteration_number):
            z_opt=update_z(z_opt-(1/gamma)*grad_z.eval(session=self.session, feed_dict={z:z_opt, u:u_opt}))
            u_opt=update_u(u_opt-(1/gamma)*grad_u.eval(session=self.session, feed_dict={z:z_opt, u:u_opt}))  
            results_hold[i]=objective_tensor.eval(session=self.session, feed_dict={z:z_opt, u:u_opt})
            if i%10==0:
                if plot:
                    self.plot_results_plus_sparse(z_opt, u_opt)
        plt.plot(range(iteration_number), results_hold)
        return(x_opt, z_opt)
        
        
        