# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 12:56:42 2021

@author: magd21
"""

#Different loss measurements!!


import numpy as np

import tensorflow as tf
import skimage
import skimage.measure

import mnist_ae_test as ae
import mnist_gan_test as gan
import mnist_vae_ns_test as vae


#%%
test_images=np.load('mnist_test_images.npy')
norm=np.linalg.norm(test_images, ord=2, axis=(1,2))**2


for num ,latent_dim in enumerate([5,6,7,8,9,10,11,12,13,14,16,20,25]):

    tf.reset_default_graph()
    sess1 = tf.Session()
    vae_none=vae.mnistVAE(latent_dim,'../mnist_vae/checkpoints_no_sigmoid_'+str(latent_dim)+'_kl_factor_0.05/checkpoints'+str(latent_dim)+'-29800')

    vae_test_enc=np.load('vae_ns_train_images_'+str(latent_dim)+'dim_kl_factor_0.05_Encodings.npy')
#    ssim_hold=np.zeros(np.shape(vae_test_enc)[0])
    psnr_hold=np.zeros(np.shape(vae_test_enc)[0])
#    l1_hold=np.zeros(np.shape(vae_test_enc)[0])

 #   nrmse_hold=np.zeros(np.shape(vae_test_enc)[0])
    
    for i in range(np.shape(vae_test_enc)[0]):
        img= vae_none.generate(vae_test_enc[i])[0,:,:]
#        ssim_hold[i]=skimage.measure.compare_ssim(img, test_images[i])
        psnr_hold[i]=skimage.measure.compare_psnr(img, test_images[i],2)
#        l1_hold[i]=np.sum(np.sum(np.abs(img-test_images[i])))
#        nrmse_hold[i]=skimage.measure.compare_nrmse(img, test_images[i])
    np.save('vae_test_'+str(latent_dim)+'dim_kl_factor_0.05_test_psnr.npy', psnr_hold)
#    np.save('vae_test_'+str(latent_dim)+'dim_kl_factor_0.05_test_ssim.npy', ssim_hold)
#    np.save('vae_test_'+str(latent_dim)+'dim_kl_factor_0.05_test_abs_error.npy', l1_hold)
    
#    np.save('vae_test_'+str(latent_dim)+'dim_kl_factor_0.05_test_nrmse.npy', nrmse_hold)

#%%



#for num ,latent_dim in enumerate([5,6,7,8,9,10,11,12,13,14,16,18,20,25]):

    tf.reset_default_graph()
    sess1 = tf.Session()
    ae_none=ae.mnistAE(latent_dim,'../ae_mnist/checkpoints_ns/checkpoints'+str(latent_dim)+'/mnist_AE_'+str(latent_dim)+'-29800')

    ae_test_enc=np.load('ae_ns_test_images_'+str(latent_dim)+'dimEncodings.npy')
#    ssim_hold=np.zeros(np.shape(ae_test_enc)[0])
    psnr_hold=np.zeros(np.shape(ae_test_enc)[0])
#    l1_hold=np.zeros(np.shape(ae_test_enc)[0])##

#    nrmse_hold=np.zeros(np.shape(vae_test_enc)[0])

    for i in range(np.shape(ae_test_enc)[0]):
        img= ae_none.generate(ae_test_enc[i])[0,:,:]
#        ssim_hold[i]=skimage.measure.compare_ssim(img, test_images[i])
        psnr_hold[i]=skimage.measure.compare_psnr(img, test_images[i],2)
#        l1_hold[i]=np.sum(np.sum(np.abs(img-test_images[i])))
#        nrmse_hold[i]=skimage.measure.compare_nrmse(img, test_images[i])

    np.save('ae_test_'+str(latent_dim)+'dim_test_psnr.npy', psnr_hold)
#    np.save('ae_test_'+str(latent_dim)+'dim_test_ssim.npy', ssim_hold)
#    np.save('ae_test_'+str(latent_dim)+'dim_test_abs_error.npy', l1_hold)

#    np.save('ae_test_'+str(latent_dim)+'dim_kl_factor_0.05_test_nrmse.npy', nrmse_hold)
  
#%%



for num ,latent_dim in enumerate([5,6,7,8,9,10,11,12,13,14,16,18,20,25]):

    tf.reset_default_graph()
    sess1 = tf.Session()
    gan_none=gan.mnistGAN(latent_dim,'../mnist_gan/checkpoints_ns/checkpoints_gan_'+str(latent_dim)+'/mnist_GAN_'+str(latent_dim)+'-19900')

    gan_test_enc=np.load('gan_ns_test_images_'+str(latent_dim)+'dimEncodings.npy')

#    ssim_hold=np.zeros(np.shape(gan_test_enc)[0])
    psnr_hold=np.zeros(np.shape(gan_test_enc)[0])
#    l1_hold=np.zeros(np.shape(gan_test_enc)[0])

#    nrmse_hold=np.zeros(np.shape(vae_test_enc)[0])

    for i in range(np.shape(gan_test_enc)[0]):
        img= gan_none.generate(gan_test_enc[i])[0,:,:]
#        ssim_hold[i]=skimage.measure.compare_ssim(img, test_images[i])
        psnr_hold[i]=skimage.measure.compare_psnr(img, test_images[i],2)
#        l1_hold[i]=np.sum(np.sum(np.abs(img-test_images[i])))
#        nrmse_hold[i]=skimage.measure.compare_nrmse(img, test_images[i])

    np.save('gan_test_'+str(latent_dim)+'dim_test_psnr.npy', psnr_hold)
#    np.save('gan_test_'+str(latent_dim)+'dim_test_ssim.npy', ssim_hold)
#    np.save('gan_test_'+str(latent_dim)+'dim_test_abs_error.npy', l1_hold)
  
 #   np.save('gan_test_'+str(latent_dim)+'dim_kl_factor_0.05_test_nrmse.npy', nrmse_hold)
