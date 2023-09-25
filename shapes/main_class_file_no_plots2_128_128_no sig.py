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


import knees2_128_128_vae_test as vae


test_images=np.load('../datasets/knee-fastMRI/knee_fastMRI_test_128_cleaned.npy')
train_images=np.load('../datasets/knee-fastMRI/knee_fastMRI_train_128_cleaned.npy')

#%%
tf.reset_default_graph()
sess = tf.InteractiveSession()
model=vae.kneesVAE(800,'./checkpoints/checkpoints_no_sigmoid/checkpoints2_128_cleaned_800_kl_factor_0.001/knees_VAE-249800',sess )




#%% Inverse problem 
import inv_prob_class as IP
#aim=np.load('aim.npy')

inv_prob=IP.invProb( model,  'tomography')


#aim=test_images[153] #The number 5 with a tick 

#aim=test_images[627] # the number 1 
aim=test_images[253]
np.random.seed(9)
#aim=model.generator(np.random.normal(0,1, (1,model.n_latent) ))

inv_prob.observe_data(aim, noise_level=0.02)
np.save('aim_knees_test_cleaned_253.npy', inv_prob.aim)
np.save('data_knees_tomography_test_cleaned_253_0.02_noise.npy', inv_prob.data)


#%%
import optimisation_class_no_plots_128 as optim

opt=optim.optimisation(inv_prob, model)








plot=opt.gd_z_regularisation_parameter(initial_z=np.random.normal(0,1,(4,opt.n_latent)), iteration_number=20, save_name='VAE2_128_128_800_0.001_z_optimisation_knees_test_cleaned_253_no_sig', save =True)
plot=opt.optim_z_sparse_regularisation_parameter(initial_z=np.random.normal(0,1,(4,opt.n_latent)), initial_u=np.zeros((4,128,128)), iteration_number=20,  save_name='VAE2_128_128_800_0.001_z_sparse_knees_test_cleaned_253_no_sig_', early_stop=False, save=True)
plot=opt.optim_x_soft_constraints_regularisation_parameter(initial_z=np.random.normal(0,1,(4,opt.n_latent)), initial_x=np.random.normal(0,1,(4,128,128)), iteration_number=50, save_name='VAE2_128_128_800_0.001_x_soft_knees_test_cleaned_253_no_sig_', early_stop=False, save=True)
plot=opt.optim_x_tik_regularisation_parameter( iteration_number=1500, save_name='x_tik_knees_test_cleaned_253_no_sig', save=True, early_stop=False)

#   
#
