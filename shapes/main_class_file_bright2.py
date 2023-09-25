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


import ae_2d_shapes_class_odl as ae
import vae_2d_shapes_class_odl as vae
import gan_2d_shapes_class_odl as gan 



import inv_prob_class as IP

import optimisation_class_no_plots as optim

test_images=np.load('./2d_shapes_test_images2_none.npy')
test_images_bright=np.load('./2d_shapes_test_images2_circle.npy')
test_masks=np.load('2d_shapes_test_images2_mask_spot_circle.npy')
#%%
tf.reset_default_graph()
sess = tf.InteractiveSession()


model=vae.shapes_VAE(10, './checkpoints/vae/checkpoints_none_10_kl_factor_0.01/2d_shapes_VAE-29800',sess)
inv_prob=IP.invProb( model, 'tomography')

np.random.seed(9)
for i in range(20):
    aim=test_images_bright[i]
    mask=[aim>=0.4]
    inv_prob.observe_data(aim, noise_level=0.05, random_seed=i)
    opt=optim.optimisation(inv_prob, model)
#    np.save('psnr_vae_hard_'+str(i)+'less_noise_bright.npy', opt.gd_z_regularisation_parameter(psnr=True, iteration_number=200, save =False))
    
#    np.save('psnr_vae_sparse_'+str(i)+'less_noise_bright.npy', opt.optim_z_sparse_regularisation_parameter(psnr=True, iteration_number=60, save =False, lambda_min=0.001))
    
#    np.save('psnr_vae_soft_'+str(i)+'less_noise_bright.npy', opt.optim_x_soft_constraints_regularisation_parameter(psnr=True, iteration_number=60, save =False))
    
#    np.save('psnr_vae_tik_'+str(i)+'less_noise_bright.npy', opt.optim_x_tik_regularisation_parameter(psnr=True, iteration_number=1000, save =False))
    
    np.save('psnr_vae_tv_'+str(i)+'less_noise_bright.npy', opt.optim_x_tv_regularisation_parameter(psnr=True, iteration_number=1000, save =False))
    
#    np.save('psnr_vae_sparse_tv_'+str(i)+'less_noise_bright.npy', opt.optim_z_sparse_tv_regularisation_parameter(psnr=True, iteration_number=60, save =False, lambda_max=0.05, lambda_min=0.001, lambda_factor=0.8))

print(elephant)

sess1 = tf.InteractiveSession()#

model=gan.shapesGAN(10, './checkpoints/wgan/checkpoints_ns_10_none/2d_shapes_wgan_10dim-29800',sess1)
inv_prob=IP.invProb( model, 'tomography')
np.random.seed(9)
for i in range(20):
    aim=test_images[i]
    mask=[aim>=0.4]

    inv_prob.observe_data(aim, noise_level=0.05, random_seed=i)
    opt=optim.optimisation(inv_prob, model)
    np.save('psnr_gan_hard_'+str(i)+'less_noise_bright.npy', opt.gd_z_regularisation_parameter(psnr=True, iteration_number=200, save=False))
    np.save('psnr_gan_sparse_'+str(i)+'less_noise_bright.npy', opt.optim_z_sparse_regularisation_parameter(psnr=True, iteration_number=60, save =False))
    np.save('psnr_gan_soft_'+str(i)+'less_noise_bright.npy', opt.optim_x_soft_constraints_regularisation_parameter(psnr=True, iteration_number=60, save =False))
#    np.save('psnr_gan_tik_'+str(i)+'less_noise_bright.npy', opt.optim_x_tik_regularisation_parameter(psnr=True, iteration_number=1000, save =False))
#    np.save('psnr_gan_tv_'+str(i)+'less_noise_bright.npy', opt.optim_x_tv_regularisation_parameter(psnr=True, iteration_number=1000, save =False))



sess2 = tf.InteractiveSession()

model=ae.shapes_AE(10, './checkpoints/ae/checkpoints_none_10/2d_shapes_AE-29800', sess2)
inv_prob=IP.invProb( model, 'tomography')
np.random.seed(9)
for i in range(20):
    aim=test_images[i]
    mask=[aim>=0.4]
    inv_prob.observe_data(aim, noise_level=0.05, random_seed=i)
    opt=optim.optimisation(inv_prob, model)
    np.save('psnr_ae_hard_'+str(i)+'less_noise_bright.npy', opt.gd_z_regularisation_parameter(psnr=True, iteration_number=200, save =False))
    np.save('psnr_ae_sparse_'+str(i)+'less_noise_bright.npy', opt.optim_z_sparse_regularisation_parameter(psnr=True, iteration_number=60, save =False))
    np.save('psnr_ae_soft_'+str(i)+'less_noise_bright.npy', opt.optim_x_soft_constraints_regularisation_parameter(psnr=True, iteration_number=60, save =False))
#    np.save('psnr_ae_tik_'+str(i)+'less_noise_bright.npy', opt.optim_x_tik_regularisation_parameter(psnr=True, iteration_number=1000, save =False))
#    np.save('psnr_ae_tv_'+str(i)+'less_noise_bright.npy', opt.optim_x_tv_regularisation_parameter(psnr=True, iteration_number=1000, save =False))



