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

test_images_bright=np.load('./2d_shapes_test_images2_circle.npy')
test_masks=np.load('2d_shapes_test_images2_mask_spot_circle.npy')

#%%
tf.reset_default_graph()
sess = tf.InteractiveSession()



model=vae.shapes_VAE(10, './checkpoints/vae/checkpoints_none_10_kl_factor_0.01/2d_shapes_VAE-29800',sess)
inv_prob=IP.invProb( model, 'tomography')


np.random.seed(9)
for i in range(100,200):
    aim=test_images_bright[i]
    inv_prob.observe_data(aim, noise_level=0.05)
    opt=optim.optimisation(inv_prob, model)
    np.save('psnr_vae_hard_'+str(i)+'fixed_bright_full_0.05.npy', opt.gd_z_regularisation_parameter(alpha_min=0.003051757812  ,alpha_max=0.0030517578125, psnr=True, iteration_number=200, save =True, save_name='psnr_vae_hard_'+str(i)+'fixed_bright_full_0.05'))
    np.save('psnr_vae_sparse_'+str(i)+'fixed_bright_full_0.05.npy', opt.optim_z_sparse_regularisation_parameter(lambda_min=0.00610351562, lambda_max=0.006103515625, mu_min=0.12, mu_max=0.125, psnr=True, iteration_number=200, save=True,save_name='psnr_vae_sparse_'+str(i)+'fixed_bright_full_0.05'))
    np.save('psnr_vae_soft_'+str(i)+'fixed_bright_full_0.05.npy', opt.optim_x_soft_constraints_regularisation_parameter(lambda_min= 0.0976562, lambda_max= 0.09765625, mu_min=0.1562, mu_max=0.15625,psnr=True, iteration_number=200, save=True,save_name='psnr_vae_soft_'+str(i)+'fixed_bright_full_0.05'))
    np.save('psnr_vae_tik_'+str(i)+'fixed_bright_full_0.05.npy', opt.optim_x_tik_regularisation_parameter( beta_min=0.0976562, beta_max=0.09765625,psnr=True, iteration_number=1000, save=True,save_name='psnr_vae_tik_'+str(i)+'fixed_bright_full_0.05'))
    np.save('psnr_vae_tv_'+str(i)+'fixed_bright_full_0.05.npy', opt.optim_x_tv_regularisation_parameter(beta_min=0.024414062, beta_max=0.0244140625, psnr=True, iteration_number=1000, save=True,save_name='psnr_vae_tv_'+str(i)+'fixed_bright_full_0.05'))
    np.save('psnr_vae_sparse_tv_'+str(i)+'fixed_bright_full_0.05.npy', opt.optim_z_sparse_tv_regularisation_parameter(lambda_min=0.00610351562, lambda_max=0.006103515625, mu_min=0.12, mu_max=0.125, psnr=True, iteration_number=200, save=True,save_name='psnr_vae_sparse_tv_'+str(i)+'fixed_bright_full_0.05'))




tf.reset_default_graph()

sess1 = tf.InteractiveSession()#

model=gan.shapesGAN(10, './checkpoints/wgan/checkpoints_ns_10_none/2d_shapes_wgan_10dim-29800',sess1)
inv_prob=IP.invProb( model, 'tomography')
np.random.seed(9)
for i in range(100,200):
    aim=test_images_bright[i]
    inv_prob.observe_data(aim, noise_level=0.05)
    opt=optim.optimisation(inv_prob, model)#
    np.save('psnr_gan_hard_'+str(i)+'fixed_bright_full_0.05.npy', opt.gd_z_regularisation_parameter(alpha_min=99 ,alpha_max=100,psnr=True, iteration_number=200, save=True,save_name='psnr_gan_hard_'+str(i)+'fixed_bright_full_0.05'))
    np.save('psnr_gan_sparse_'+str(i)+'fixed_bright_full_0.05.npy', opt.optim_z_sparse_regularisation_parameter(lambda_min=0.0122070312, lambda_max=0.01220703125, mu_min=0.01562, mu_max=0.015625,psnr=True, iteration_number=200, save=True,save_name='psnr_gan_sparse_'+str(i)+'fixed_bright_full_0.05'))
    np.save('psnr_gan_soft_'+str(i)+'fixed_bright_full_0.05.npy', opt.optim_x_soft_constraints_regularisation_parameter(lambda_min=0.024414062, lambda_max=0.0244140625, mu_min= 0.1562, mu_max= 0.15625,psnr=True, iteration_number=200, save=True, save_name='psnr_gan_soft_'+str(i)+'fixed_bright_full_0.05'))
#    np.save('psnr_gan_tik_'+str(i)+'fixed_bright.npy', opt.optim_x_tik_regularisation_parameter(beta_min=0.04882812, beta_max=0.048828125,psnr=True, iteration_number=1000, save=True,save_name='psnr_gan_tik_'+str(i)+'fixed_bright'))
#    np.save('psnr_gan_tv_'+str(i)+'fixed_bright.npy', opt.optim_x_tv_regularisation_parameter(beta_min=0.00610351562, beta_max=0.006103515625,psnr=True, iteration_number=1000, save=True,save_name='psnr_gan_tv_'+str(i)+'fixed_bright'))



tf.reset_default_graph()

sess2 = tf.InteractiveSession()

model=ae.shapes_AE(10, './checkpoints/ae/checkpoints_none_10/2d_shapes_AE-29800', sess2)
inv_prob=IP.invProb( model, 'tomography')
np.random.seed(9)
for i in range(100,200):
    aim=test_images_bright[i]
    inv_prob.observe_data(aim, noise_level=0.05)
    opt=optim.optimisation(inv_prob, model)
    np.save('psnr_ae_hard_'+str(i)+'fixed_bright_full_0.05.npy', opt.gd_z_regularisation_parameter(alpha_min=0.04882812  ,alpha_max=0.048828125,psnr=True, iteration_number=200, save=True, save_name='psnr_ae_hard_'+str(i)+'fixed_bright_full_0.05'))
    np.save('psnr_ae_sparse_'+str(i)+'fixed_bright_full_0.05.npy', opt.optim_z_sparse_regularisation_parameter(lambda_min=0.7812, lambda_max=0.78125, mu_min=0.12, mu_max=0.125, psnr=True, iteration_number=200, save=True,save_name='psnr_ae_sparse_'+str(i)+'fixed_bright_full_0.05'))
    np.save('psnr_ae_soft_'+str(i)+'fixed_bright_full_0.05.npy', opt.optim_x_soft_constraints_regularisation_parameter(lambda_min=0.39062, lambda_max=0.390625, mu_min=0.62, mu_max=0.625, psnr=True, iteration_number=200, save=True,save_name='psnr_ae_soft_'+str(i)+'fixed_bright_full_0.05'))
#    np.save('psnr_ae_tik_'+str(i)+'fixed_bright.npy', opt.optim_x_tik_regularisation_parameter(beta_min=0.04882812, beta_max=0.048828125,psnr=True, iteration_number=1000, save=True,save_name='psnr_ae_tik_'+str(i)+'fixed_bright'))
#    np.save('psnr_ae_tv_'+str(i)+'fixed_bright.npy', opt.optim_x_tv_regularisation_parameter(beta_min=0.00610351562, beta_max=0.006103515625,psnr=True, iteration_number=1000, save=True,save_name='psnr_ae_tv_'+str(i)+'fixed_bright'))



