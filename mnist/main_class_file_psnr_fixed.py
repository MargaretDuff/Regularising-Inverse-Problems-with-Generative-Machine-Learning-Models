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
import mnist_gan_test_class as gan 

import inv_prob_class as IP

import optimisation_class_no_plots as optim

test_images=np.load('./mnist_test_images.npy')
#%%
#tf.reset_default_graph()
sess = tf.InteractiveSession()


#model=vae.mnistVAE(8,'./checkpoints/vae/checkpoints_no_sigmoid_8_kl_factor_0.05/checkpoints8-29800', sess)
#inv_prob=IP.invProb( model, 'compressed_sensing', kernel_width=8, dim_comp=150)

#np.random.seed(9)
#for i in range(100,200):
#    aim=test_images[i]
#    inv_prob.observe_data(aim, noise_level=0.1, random_seed=i)
#    np.save('aim_'+str(i)+'.npy', aim)
#    np.save('data_'+str(i)+'compressed150_0.1_noise.npy', inv_prob.data)
#    opt=optim.optimisation(inv_prob, model)
#    np.save('psnr_vae_hard_'+str(i)+'compressed150_fixed.npy', opt.gd_z_regularisation_parameter(alpha_min=0.03051757812  ,alpha_max=0.030517578125, psnr=True, iteration_number=200, save =True, save_name='psnr_vae_hard_'+str(i)+'compressed150_fixed'))
#    np.save('psnr_vae_sparse_'+str(i)+'compressed150_fixed.npy', opt.optim_z_sparse_regularisation_parameter(lambda_min=31, lambda_max=31.25, mu_min=0.62, mu_max=0.625, psnr=True, iteration_number=200, save=True,save_name='psnr_vae_sparse_'+str(i)+'compressed150_fixed'))
#    np.save('psnr_vae_soft_'+str(i)+'compressed150_fixed.npy', opt.optim_x_soft_constraints_regularisation_parameter(lambda_min=31, lambda_max=31.25, mu_min=0.039062, mu_max=0.0390625,psnr=True, iteration_number=200, save=True,save_name='psnr_vae_soft_'+str(i)+'compressed150_fixed'))
#    np.save('psnr_vae_tik_'+str(i)+'compressed150_fixed.npy', opt.optim_x_tik_regularisation_parameter( beta_min=7.812, beta_max=7.8125,psnr=True, iteration_number=1000, save=True,save_name='psnr_vae_tik_'+str(i)+'compressed150_fixed'))
#    np.save('psnr_vae_tv_'+str(i)+'compressed150_fixed.npy', opt.optim_x_tv_regularisation_parameter(beta_min=0.00610351562, beta_max=0.006103515625, psnr=True, iteration_number=1000, save=True,save_name='psnr_vae_tv_'+str(i)+'compressed150_fixed'))


#tf.reset_default_graph()

#sess1 = tf.InteractiveSession()#

#model=gan.mnistGAN(8,'./checkpoints/wgan/checkpoints_gan_8/mnist_GAN_8-19900' ,sess1)
#inv_prob=IP.invProb( model, 'compressed_sensing',dim_comp=150, kernel_width=8)
#np.random.seed(9)
#for i in range(100,200):
#    aim=test_images[i]
#    inv_prob.observe_data(aim, noise_level=0.1, random_seed=i)
#    opt=optim.optimisation(inv_prob, model)
#    np.save('psnr_gan_hard_'+str(i)+'compressed150_fixed.npy', opt.gd_z_regularisation_parameter(alpha_min=499  ,alpha_max=500,psnr=True, iteration_number=200, save=True,save_name='psnr_gan_hard_'+str(i)+'compressed150_fixed'))
#    np.save('psnr_gan_sparse_'+str(i)+'compressed150_fixed.npy', opt.optim_z_sparse_regularisation_parameter(lambda_min=31.2, lambda_max=31.25, mu_min=9, mu_max=10, psnr=True, iteration_number=200, save=True,save_name='psnr_gan_sparse_'+str(i)+'compressed150_fixed'))
#    np.save('psnr_gan_soft_'+str(i)+'compressed150_fixed.npy', opt.optim_x_soft_constraints_regularisation_parameter(lambda_min=31.2, lambda_max=31.25, mu_min=0.1562, mu_max=0.15625,psnr=True, iteration_number=200, save=True, save_name='psnr_gan_soft_'+str(i)+'compressed150_fixed'))
#    np.save('psnr_gan_tik_'+str(i)+'compressed150_fixed.npy', opt.optim_x_tik_regularisation_parameter(beta_min=7.812, beta_max=7.8125,psnr=True, iteration_number=1000, save=True,save_name='psnr_gan_tik_'+str(i)+'compressed150_fixed'))
#    np.save('psnr_gan_tv_'+str(i)+'compressed150_fixed.npy', opt.optim_x_tv_regularisation_parameter(beta_min=0.00610351562, beta_max=0.006103515625,psnr=True, iteration_number=1000, save=True,save_name='psnr_gan_tv_'+str(i)+'compressed150_fixed'))



tf.reset_default_graph()

sess2 = tf.InteractiveSession()

model=ae.mnist_AE(8,0,'./checkpoints/ae/checkpoints8/mnist_AE_8-29800' ,sess2)
inv_prob=IP.invProb( model, 'compressed_sensing',dim_comp=150, kernel_width=8)
np.random.seed(9)
for i in range(100,200):
    aim=test_images[i]
    inv_prob.observe_data(aim, noise_level=0.1, random_seed=i)
    opt=optim.optimisation(inv_prob, model)
    np.save('psnr_ae_hard_'+str(i)+'compressed150_fixed.npy', opt.gd_z_regularisation_parameter(alpha_min=0.03051757812 ,alpha_max=0.030517578125,psnr=True, iteration_number=200, save=True, save_name='psnr_ae_hard_'+str(i)+'compressed150_fixed'))
    np.save('psnr_ae_sparse_'+str(i)+'compressed150_fixed.npy', opt.optim_z_sparse_regularisation_parameter(lambda_min=31.2, lambda_max=31.25, mu_min=0.039062, mu_max=0.0390625,psnr=True, iteration_number=200, save=True,save_name='psnr_ae_sparse_'+str(i)+'compressed150_fixed'))
    np.save('psnr_ae_soft_'+str(i)+'compressed150_fixed.npy', opt.optim_x_soft_constraints_regularisation_parameter(lambda_min=31.2, lambda_max=31.25, mu_min=0.039062, mu_max=0.0390625,psnr=True, iteration_number=200, save=True,save_name='psnr_ae_soft_'+str(i)+'compressed150_fixed'))
    np.save('psnr_ae_tik_'+str(i)+'compressed150_fixed.npy', opt.optim_x_tik_regularisation_parameter(beta_min=7.812, beta_max=7.8125,psnr=True, iteration_number=1000, save=True,save_name='psnr_ae_tik_'+str(i)+'compressed150_fixed'))
    np.save('psnr_ae_tv_'+str(i)+'compressed150_fixed.npy', opt.optim_x_tv_regularisation_parameter(beta_min=0.00610351562, beta_max=0.006103515625,psnr=True, iteration_number=1000, save=True,save_name='psnr_ae_tv_'+str(i)+'compressed150_fixed'))



