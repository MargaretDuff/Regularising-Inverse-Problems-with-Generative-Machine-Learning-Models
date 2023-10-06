# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 12:52:46 2020

@author: marga
"""

import numpy as np
from scipy import stats
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2


import generative_model as test_class
import mnist_ae_test as ae
import mnist_gan_test_class as gan
import mnist_vae_test as vae

test_images=np.load('../mnist_test_images.npy')

tf.reset_default_graph()
sess1=tf.InteractiveSession()
ae_test=ae.mnistAE(8, 0.0,'../AE_checkpoints8_0.0/mnist_AE_8_0.0-29800',sess1)

tf.reset_default_graph()
sess2=tf.InteractiveSession()
vae_test=vae.mnistVAE(8, '../VAE_Testing/checkpoints8/mnist_VAE-29800', sess2)

tf.reset_default_graph()
sess3=tf.InteractiveSession()

gan_test=gan.mnistGAN(16, '../GAN_checkpoints16/mnist_GAN_16-39900', sess3)

#%%#%% Image projections
R_normed=ae_test.random_projection_image(test_images, save_name='AE_mnist_8dim', z_variation=30)
#random_projection_image(self, image_set, save_name='Test', z_variation=1, R_normed=None)
vae_test.random_projection_image(test_images, save_name='VAE_mnist_8dim', R_normed=R_normed)
gan_test.random_projection_image(test_images, save_name='GAN_mnist_16dim', R_normed=R_normed)




#%%
#Image reconstuction contour plots
#np.save('Initialisation8.npy', np.random.normal(0,1,(1,8)))
z8=np.load('../Initialisation8.npy')
#np.save('Initialisation16.npy', np.random.normal(0,1,(1,16)))
z16=np.load('../Initialisation16.npy')
vae_test.observationLossRandomPlots(vae_test.generate(z8),z=z8, save=True, save_name='VAE_mnist_8dim_generated')
ae_test.observationLossRandomPlots(ae_test.generate(z8), z=z8,minimum=-20, maximum=20, save=True, save_name='AE_mnist_8dim_generated')
gan_test.observationLossRandomPlots(gan_test.generate(z16),z=z16,  save=True, save_name='GAN_mnist_16dim_generated')

#%%

#Far from reconstructions
for i in range(6):
    cv2.imwrite('ae_8_dim_mnist_generate_far_from_prior_'+str(i)+'.png', ae_test.generate(50*np.random.normal(0,1, ae_test.n_latent))[0,:,:]*256)
    cv2.imwrite('gan_16_dim_mnist_generate_far_from_prior_'+str(i)+'.png', gan_test.generate(5*np.random.normal(0,1, gan_test.n_latent))[0,:,:]*256)
    cv2.imwrite('vae_8_dim_mnist_generate_far_from_prior_'+str(i)+'.png', vae_test.generate(5*np.random.normal(0,1, vae_test.n_latent))[0,:,:]*256)
#%%

ae_test.emd_images(test_images, save_name='AE_mnist_8dim')
vae_test.emd_images(test_images, save_name='VAE_mnist_8dim')
gan_test.emd_images(test_images, save_name='GAN_mnist_16dim')



#%% Interpolations
img_no1=650
img_no2=2450
ae_test.interpolation(ae_test.encode(test_images[img_no1]), ae_test.encode(test_images[img_no2]),intervals=101, save=True, save_name='AE_8dim')
vae_test.interpolation(vae_test.encode(test_images[img_no1]), vae_test.encode(test_images[img_no2]),intervals=101, save=True, save_name='VAE_8dim')
gan_test.interpolation(gan_test.encode(test_images[img_no1]), gan_test.encode(test_images[img_no2]),intervals=101, save=True, save_name='GAN_16dim')



