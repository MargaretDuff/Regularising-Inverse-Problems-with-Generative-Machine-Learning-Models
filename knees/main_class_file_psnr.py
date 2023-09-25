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

test_images=np.load('../datasets/knee-fastMRI/knee_fastMRI_test.npy')
#%%
#tf.reset_default_graph()
sess = tf.InteractiveSession()


#model=vae.mnistVAE(8,'./checkpoints/vae/checkpoints_no_sigmoid_8_kl_factor_0.05/checkpoints8-29800', sess)
#inv_prob=IP.invProb( model, 'convolution')

#np.random.seed(9)
#for i in range(20):
#    aim=test_images[i]
#    inv_prob.observe_data(aim, noise_level=0.1)
#    opt=optim.optimisation(inv_prob, model)
#    np.save('psnr_vae_hard_'+str(i)+'.npy', opt.gd_z_regularisation_parameter(psnr=True, iteration_number=200, save =False))
#    np.save('psnr_vae_sparse_'+str(i)+'.npy', opt.optim_z_sparse_regularisation_parameter(psnr=True, iteration_number=200, save =False))
#    np.save('psnr_vae_soft_'+str(i)+'.npy', opt.optim_x_soft_constraints_regularisation_parameter(psnr=True, iteration_number=200, save =False))
#    np.save('psnr_vae_tik_'+str(i)+'.npy', opt.optim_x_tik_regularisation_parameter(psnr=True, iteration_number=1000, save =False))
#    np.save('psnr_vae_tv_'+str(i)+'.npy', opt.optim_x_tv_regularisation_parameter(psnr=True, iteration_number=1000, save =False))



sess1 = tf.InteractiveSession()

model=gan.mnistGAN(8,'./checkpoints/wgan/checkpoints_gan_8/mnist_GAN_8-19900' ,sess1)
inv_prob=IP.invProb( model, 'convolution')
np.random.seed(9)
for i in range(20):
    aim=test_images[i]
    inv_prob.observe_data(aim, noise_level=0.1)
    opt=optim.optimisation(inv_prob, model)
    np.save('psnr_gan_hard_'+str(i)+'.npy', opt.gd_z_regularisation_parameter(psnr=True, iteration_number=200, save=False))
    np.save('psnr_gan_sparse_'+str(i)+'.npy', opt.optim_z_sparse_regularisation_parameter(psnr=True, iteration_number=200, save =False))
    np.save('psnr_gan_soft_'+str(i)+'.npy', opt.optim_x_constraints_regularisation_soft_parameter(psnr=True, iteration_number=200, save =False))
    np.save('psnr_gan_tik_'+str(i)+'.npy', opt.optim_x_tik_regularisation_parameter(psnr=True, iteration_number=1000, save =False))
    np.save('psnr_gan_tv_'+str(i)+'.npy', opt.optim_x_tv_regularisation_parameter(psnr=True, iteration_number=1000, save =False))




#sess2 = tf.InteractiveSession()

#model=ae.mnistAE(8,0,'./checkpoints/ae/checkpoints8/mnist_AE_8-29800',sess2 )
#inv_prob=IP.invProb( model, 'convolution')

#np.random.seed(9)
#for i in range(20):
#    aim=test_images[i]
#    inv_prob.observe_data(aim, noise_level=0.1)
#    opt=optim.optimisation(inv_prob, model)
#    np.save('psnr_ae_hard_'+str(i)+'.npy', opt.gd_z_regularisation_parameter(psnr=True, iteration_number=200, save =False))
#    np.save('psnr_ae_sparse_'+str(i)+'.npy', opt.optim_z_sparse_regularisation_parameter(psnr=True, iteration_number=200, save =False))
#    np.save('psnr_ae_soft_'+str(i)+'.npy', opt.optim_x_soft_constraints_regularisation_parameter(psnr=True, iteration_number=200, save =False))
#    np.save('psnr_ae_tik_'+str(i)+'.npy', opt.optim_x_tik_regularisation_parameter(psnr=True, iteration_number=1000, save =False))
#    np.save('psnr_ae_tv_'+str(i)+'.npy', opt.optim_x_tv_regularisation_parameter(psnr=True, iteration_number=1000, save =False))



