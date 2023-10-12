# -*- coding: utf-8 -*-

import optimisation_class_no_plots_128 as optim
import inv_prob_class as IP
import knees2_128_128_vae_test as vae
import tensorflow as tf
import numpy as np
test_images = np.load('./datasets/knee_fastMRI_test_128_cleaned.npy')


tf.reset_default_graph()
sess = tf.InteractiveSession()

# Import the VAE model and load from a checkpoint
checkpoint_file = 'USER DEFINED PATH'
model = vae.kneesVAE(800, checkpoint_file, sess)

# Set up the inverse problem
inv_prob = IP.invProb(model,  'tomography')
aim = test_images[253]

# Create noisy observed data
np.random.seed(9)
inv_prob.observe_data(aim, noise_level=0.02)
np.save('aim_knees_test_cleaned_253.npy', inv_prob.aim)
np.save('data_knees_tomography_test_cleaned_253_0.02_noise.npy', inv_prob.data)


#Load the optimisation class 
opt = optim.optimisation(inv_prob, model)

#Test the 4 different optimisation methods 
plot = opt.gd_z_regularisation_parameter(initial_z=np.random.normal(
    0, 1, (4, opt.n_latent)), iteration_number=200, save_name='VAE2_128_128_800_0.001_z_optimisation_knees_test_cleaned_253_no_sig', save=True)
plot = opt.optim_z_sparse_regularisation_parameter(initial_z=np.random.normal(0, 1, (4, opt.n_latent)), initial_u=np.zeros(
    (4, 128, 128)), iteration_number=200,  save_name='VAE2_128_128_800_0.001_z_sparse_knees_test_cleaned_253_no_sig_', early_stop=False, save=True)
plot = opt.optim_x_soft_constraints_regularisation_parameter(initial_z=np.random.normal(0, 1, (4, opt.n_latent)), initial_x=np.random.normal(
    0, 1, (4, 128, 128)), iteration_number=200, save_name='VAE2_128_128_800_0.001_x_soft_knees_test_cleaned_253_no_sig_', early_stop=False, save=True)
plot = opt.optim_x_tik_regularisation_parameter(
    iteration_number=3000, save_name='x_tik_knees_test_cleaned_253_no_sig', save=True, early_stop=False)

#
#
