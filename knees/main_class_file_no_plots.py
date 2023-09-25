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


import knees7_vae_test as vae


test_images=np.load('../datasets/knee-fastMRI/knee_fastMRI_test.npy')
train_images=np.load('../datasets/knee-fastMRI/knee_fastMRI_train.npy')

#%%
tf.reset_default_graph()
sess = tf.InteractiveSession()
model=vae.kneesVAE(800,'./checkpoints_no_sigmoid/checkpoints7_800_kl_factor_0.05/knees_VAE-399800',sess )




#%% Inverse problem 
import inv_prob_class as IP
#aim=np.load('aim.npy')

inv_prob=IP.invProb( model,  'tomography')


#aim=test_images[153] #The number 5 with a tick 

aim=test_images[378] # the number 1 
#aim=train_images[638]
np.random.seed(9)
#aim=model.generator(np.random.normal(0,1, (1,model.n_latent) ))

inv_prob.observe_data(aim,  noise_level=0.02)
np.save('aim_knees_test_395.npy', inv_prob.aim)
np.save('data_knees_tomography_test_395_0.02_noise.npy', inv_prob.data)


#%%
import optimisation_class_no_plots as optim

opt=optim.optimisation(inv_prob, model)


#opt.gd_backtracking_z(1, model.encoder(inv_prob.data))

#z,u=opt.optim_z_sparse_deviations(1,0.1, model.encoder(inv_prob.data), 0.01*np.random.normal(0,1,(80,80)), 100)

#opt.optim_x_soft_constraints(1, 1, model.encoder(inv_prob.data), model.generator(model.encoder(inv_prob.data)) ,200)


#%%
#plot=opt.gd_z_regularisation_parameter(initial_z=model.encoder(inv_prob.data), iteration_number=20, save_name='VAE7_800_0.05_z_optimisation_knees_test_353_', save =True)
#plot=opt.optim_z_sparse_regularisation_parameter(initial_z=model.encoder(inv_prob.data), initial_u=np.zeros((1,80,80)), iteration_number=20,  save_name='VAE7_800_0.05_z_sparse_knees_test_353_', early_stop=False, save=True)
#plot=opt.optim_x_soft_constraints_regularisation_parameter(initial_z=model.encoder(inv_prob.data), initial_x=np.random.normal(0,1,(1,80,80)), iteration_number=20, save_name='VAE7_800_0.05_x_soft_knees_test_627_', e353y_stop=False, save=True)
#plot=opt.optim_x_tik_regularisation_parameter( iteration_number=1500, save_name='x_tik_knees_test_353', save=True, early_stop=False)



plot=opt.gd_z_regularisation_parameter(initial_z=np.random.normal(0,1,(4,opt.n_latent)), iteration_number=20, save_name='VAE7_800_0.05_z_optimisation_knees_test_395_', save =True)
plot=opt.optim_z_sparse_regularisation_parameter(initial_z=np.random.normal(0,1,(4,opt.n_latent)), initial_u=np.zeros((4,80,80)), iteration_number=20,  save_name='VAE7_800_0.05_z_sparse_knees_test_395_', early_stop=False, save=True)
plot=opt.optim_x_soft_constraints_regularisation_parameter(initial_z=np.random.normal(0,1,(4,opt.n_latent)), initial_x=np.random.normal(0,1,(4,80,80)), iteration_number=20, save_name='VAE7_800_0.05_x_soft_knees_test_395_', early_stop=False, save=True)
plot=opt.optim_x_tik_regularisation_parameter( iteration_number=1500, save_name='x_tik_knees_test_395', save=True, early_stop=False)

#   
#
