# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 16:57:31 2020

@author: magd21
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 12:22:01 2020

@author: marga
"""

import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt


import mnist_ae_test as ae
import mnist_vae_test as vae
import mnist_gan_test as gan 


#train_images=np.load('../mnist_train_images.npy')
test_images=np.load('../mnist_test_images.npy')
# for i in range(150,160):
#     plt.figure()
#     plt.imshow(test_images[i])


#%%
tf.reset_default_graph()
sess = tf.InteractiveSession()

model=ae.mnistAE(8,0,'../../ae_mnist/checkpoints_ns/checkpoints8/mnist_AE_8-29800',sess )
## test.elephant()
#
#
##%% Inverse problem 
import inv_prob_class as IP
##aim=np.load('aim.npy')
#
inv_prob=IP.invProb( model, 'convolution')
#
#
aim=test_images[533]
#
#aim=test_images[2] # the number 1 
#np.random.seed(9)
##aim=model.generator(np.random.normal(0,1, (1,model.n_latent) ))
#

np.random.seed(9)
inv_prob.observe_data(aim, noise_level=0.1)
np.save('aim_mnist_533.npy', inv_prob.aim)
np.save('data_mnist_convolution_noise_0.1_test_533.npy', inv_prob.data)
#
#
##%%
import optimisation_class_no_plots as optim
#
opt=optim.optimisation(inv_prob, model)
#

for i in range(10):
   plot=opt.gd_z_regularisation_parameter( iteration_number=30, save_name='AE_8_dim_no_sigmoid_z_optimisation_533_iter_'+str(i), save =True)
   plot=opt.optim_z_sparse_regularisation_parameter( iteration_number=30, mu_min=0.49, mu_max=0.5,  save_name='AE_8_dim_no_sigmoid_z_sparse_533_iter_'+str(i), early_stop=False, save=True)
   plot=opt.optim_x_soft_constraints_regularisation_parameter( iteration_number=30, mu_min=0.49, mu_max=0.5, save_name='AE_8_dim_no_sigmoid_x_soft_533_iter_'+str(i), early_stop=False, save=True)
   

#plot=opt.gd_z_regularisation_parameter(initial_z=np.random.normal(0,1,(4,model.n_latent)), iteration_number=30, save_name='AE_8_dim_no_sigmoid_z_optimisation_398_', save =True)
#plot=opt.optim_z_sparse_regularisation_parameter(initial_z=np.random.normal(0,1,(4,model.n_latent)), initial_u=np.zeros((4,28,28)), iteration_number=30,  save_name='AE_8_dim_no_sigmoid_z_sparse_398_', early_stop=False, save=True)
#plot=opt.optim_x_soft_constraints_regularisation_parameter(initial_z=np.random.normal(0,1,(4,model.n_latent)), initial_x=np.random.normal(0,1,(4,28,28)), iteration_number=30, save_name='AE_8_dim_no_sigmoid_x_soft_398', early_stop=False, save=True)



plot=opt.optim_x_tik_regularisation_parameter( iteration_number=1500, save_name='x_tik_533', save=True, early_stop=False)
##       
#

#%%
tf.reset_default_graph()
sess = tf.InteractiveSession()
#model=vae.mnistVAE(8,'../checkpoints_no_sigmoid/vae_8_dim/checkpoints8-29800',sess)

model=vae.mnistVAE(8,'../../mnist_vae/checkpoints_no_sigmoid_8_kl_factor_0.05/checkpoints8-29800',sess)


#%%
import inv_prob_class as IP
#aim=np.load('aim.npy')

inv_prob=IP.invProb( model, 'convolution')


#aim=test_images[153] #The number 5 with a tick 
#aim=test_images[128]
#aim=test_images[2] # the number 1 
np.random.seed(9)
#aim=model.generator(np.random.normal(0,1, (1,model.n_latent) ))

inv_prob.observe_data(aim, noise_level=0.1)
#np.save('aim_mnist_convolution_128.npy', inv_prob.aim)
#np.save('data_mnist_convolution_128.npy', inv_prob.data)

#%%
import optimisation_class_no_plots as optim

opt=optim.optimisation(inv_prob, model)

for i in range(10):
   plot=opt.gd_z_regularisation_parameter( iteration_number=30, save_name='VAE_8_dim_no_sigmoid_z_optimisation_533_iter_'+str(i), save =True)
   plot=opt.optim_z_sparse_regularisation_parameter( iteration_number=30, mu_min=0.49, mu_max=0.5,  save_name='VAE_8_dim_no_sigmoid_z_sparse_533_iter_'+str(i), early_stop=False, save=True)
   plot=opt.optim_x_soft_constraints_regularisation_parameter( iteration_number=30, mu_min=0.49, mu_max=0.5, save_name='VAE_8_dim_no_sigmoid_x_soft_533_iter_'+str(i), early_stop=False, save=True)


#plot=opt.gd_z_regularisation_parameter(initial_z=np.random.normal(0,1,(4,model.n_latent)), iteration_number=30, save_name='VAE_8_dim_kl_factor_0.05_no_sigmoid_z_optimisation_398_', save =True)
#plot=opt.optim_z_sparse_regularisation_parameter(initial_z=np.random.normal(0,1,(4,model.n_latent)), initial_u=np.zeros((4,28,28)), iteration_number=30,  save_name='VAE_8_dim_kl_factor_0.05_no_sigmoid_z_sparse_398_', early_stop=False, save=True)
#plot=opt.optim_x_soft_constraints_regularisation_parameter(initial_z=np.random.normal(0,1,(4,model.n_latent)), initial_x=np.random.normal(0,1,(4,28,28)), iteration_number=30, save_name='VAE_8_dim_kl_factor_0.05_no_sigmoid_x_soft_398', early_stop=False, save=True)



#%%
tf.reset_default_graph()
sess = tf.InteractiveSession()
#model=gan.mnistGAN(8,'../checkpoints_no_sigmoid/gan_8_dim/mnist_GAN_8-19900',sess )

model=gan.mnistGAN(8,'../../mnist_gan/checkpoints_ns/checkpoints_gan_8/mnist_GAN_8-19900',sess )

#%%
import inv_prob_class as IP
#aim=np.load('aim.npy')

inv_prob=IP.invProb( model, 'convolution')


#aim=test_images[153] #The number 5 with a tick 

#aim=test_images[2] # the number 1 
np.random.seed(9)
#aim=model.generator(np.random.normal(0,1, (1,model.n_latent) ))

inv_prob.observe_data(aim, noise_level=0.1)
#np.save('aim_mnist_convolution_one.npy', inv_prob.aim)
#np.save('data_mnist_convolution_one.npy', inv_prob.data)
#%%
import optimisation_class_no_plots as optim

opt=optim.optimisation(inv_prob, model)

for i in range(10):
   plot=opt.gd_z_regularisation_parameter( iteration_number=30, save_name='GAN_8_dim_no_sigmoid_z_optimisation_533_iter_'+str(i), save =True)
   plot=opt.optim_z_sparse_regularisation_parameter( iteration_number=30, mu_min=0.49, mu_max=0.5,  save_name='GAN_8_dim_no_sigmoid_z_sparse_533_iter_'+str(i), early_stop=False, save=True)
   plot=opt.optim_x_soft_constraints_regularisation_parameter( iteration_number=30, mu_min=0.49, mu_max=0.5, save_name='GAN_8_dim_no_sigmoid_x_soft_533_iter_'+str(i), early_stop=False, save=True)


#plot=opt.gd_z_regularisation_parameter(initial_z=np.random.normal(0,1,(4,model.n_latent)), iteration_number=30, save_name='GAN_8_dim_no_sigmoid_z_optimisation_398_', save =True)
#plot=opt.optim_z_sparse_regularisation_parameter(initial_z=np.random.normal(0,1,(4,model.n_latent)), initial_u=np.zeros((4,28,28)), iteration_number=30,  save_name='GAN_8_dim_no_sigmoid_z_sparse_398_', early_stop=False, save=True)
#plot=opt.optim_x_soft_constraints_regularisation_parameter(initial_z=np.random.normal(0,1,(4,model.n_latent)), initial_x=np.random.normal(0,1,(4,28,28)), iteration_number=30, save_name='GAN_8_dim_no_sigmoid_x_soft_398', early_stop=False, save=True)

#       

