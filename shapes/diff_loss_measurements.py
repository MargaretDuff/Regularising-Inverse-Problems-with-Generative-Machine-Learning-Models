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
import vae_2d_shapes_class as vae
import ae_2d_shapes_class as ae
import wgan_2d_shapes_class as gan

#%%
test_images=np.load('../../datasets/2d_shapes/2d_shapes_test_images_none.npy')
norm=np.linalg.norm(test_images, ord=2, axis=(1,2))**2


#for num ,latent_dim in enumerate([5,6,7,8,9,10,11,12,13,14,16,18,20,25]):

#    tf.reset_default_graph()
#    sess1 = tf.Session()
#    vae_none=vae.shapes_VAE(latent_dim,'../../2d_shapes_vae/checkpoints_no_sigmoid/checkpoints_none_'+str(latent_dim)+'_kl_factor_0.01/2d_shapes_VAE-29800',sess1)

#    vae_test_enc=np.load('vae_test2_none_train_ns_'+str(latent_dim)+'dim_kl_factor_0.01_noneEncodings.npy')
#    ssim_hold=np.zeros(np.shape(vae_test_enc)[0])
#    psnr_hold=np.zeros(np.shape(vae_test_enc)[0])
#    nrmse=np.zeros(np.shape(vae_test_enc)[0])
#    l1_hold=np.zeros(np.shape(vae_test_enc)[0])
    
#    for i in range(np.shape(vae_test_enc)[0]):
#        img= vae_none.generate(vae_test_enc[i])[0,:,:]
#        ssim_hold[i]=skimage.measure.compare_ssim(img, test_images[i])
#        psnr_hold[i]=skimage.measure.compare_psnr(img, test_images[i],2)
#        l1_hold[i]=np.sum(np.sum(np.abs(img-test_images[i])))
#        nrmse[i]=skimage.measure.compare_nrmse(img, test_images[i])


#    np.save('vae_test_'+str(latent_dim)+'dim_kl_factor_0.01_test_psnr.npy', psnr_hold)
#    np.save('vae_test_'+str(latent_dim)+'dim_kl_factor_0.01_test_ssim.npy', ssim_hold)
#    np.save('vae_test_'+str(latent_dim)+'dim_kl_factor_0.01_test_abs_error.npy', l1_hold)
#        np.save('vae_test_'+str(latent_dim)+'dim_kl_factor_0.01_test_nrmse.npy', nrmse)

#%%



#for num ,latent_dim in enumerate([5,6,7,8,9,10,11,12,13,14,16,18,20,25]):

#    tf.reset_default_graph()
#    sess1 = tf.Session()
#    ae_none=ae.shapes_AE(latent_dim,'../../2d_shapes_ae/checkpoints_no_sigmoid/checkpoints_none_'+str(latent_dim)+'/2d_shapes_AE-29800',sess1)

#    ae_test_enc=np.load('ae_test2_none_train_ns_'+str(latent_dim)+'dim_noneEncodings.npy')
#    ssim_hold=np.zeros(np.shape(ae_test_enc)[0])
#    psnr_hold=np.zeros(np.shape(ae_test_enc)[0])
#    l1_hold=np.zeros(np.shape(ae_test_enc)[0])
#    nrmse=np.zeros(np.shape(ae_test_enc)[0])

#    for i in range(np.shape(ae_test_enc)[0]):
#        img= ae_none.generate(ae_test_enc[i])[0,:,:]
#        ssim_hold[i]=skimage.measure.compare_ssim(img, test_images[i])
#        psnr_hold[i]=skimage.measure.compare_psnr(img, test_images[i],2)
#        l1_hold[i]=np.sum(np.sum(np.abs(img-test_images[i])))
#        nrmse[i]=skimage.measure.compare_nrmse(img, test_images[i])
#    np.save('ae_test_'+str(latent_dim)+'dim_kl_factor_0.01_test_psnr.npy', psnr_hold)
#    np.save('ae_test_'+str(latent_dim)+'dim_kl_factor_0.01_test_ssim.npy', ssim_hold)
#    np.save('ae_test_'+str(latent_dim)+'dim_kl_factor_0.01_test_abs_error.npy', l1_hold)
#        np.save('ae_test_'+str(latent_dim)+'dim_kl_factor_0.01_test_nrmse.npy', nrmse)

#%%



for num ,latent_dim in enumerate([5,6,7,8,9,10,11,12,13,14,16,18,20,25]):

    tf.reset_default_graph()
    sess1 = tf.Session()
    gan_none=gan.shapesGAN(latent_dim,'../../2d_shapes_wgan/checkpoints_ns_'+str(latent_dim)+'_none/2d_shapes_wgan_'+str(latent_dim)+'dim-29800',sess1)

    gan_test_enc=np.load('gan_test2_none_train_ns_'+str(latent_dim)+'dim_noneEncodings.npy')

#    ssim_hold=np.zeros(np.shape(gan_test_enc)[0])
    psnr_hold=np.zeros(np.shape(gan_test_enc)[0])
#    l1_hold=np.zeros(np.shape(gan_test_enc)[0])

#    nrmse=np.zeros(np.shape(gan_test_enc)[0])

    for i in range(np.shape(gan_test_enc)[0]):
        img= gan_none.generate(gan_test_enc[i])[0,:,:]
#        ssim_hold[i]=skimage.measure.compare_ssim(img, test_images[i])
        psnr_hold[i]=skimage.measure.compare_psnr(img, test_images[i],2)
#        l1_hold[i]=np.sum(np.sum(np.abs(img-test_images[i])))

#        nrmse[i]=skimage.measure.compare_nrmse(img, test_images[i])

    np.save('gan_test_'+str(latent_dim)+'dim_kl_factor_0.01_test_psnr.npy', psnr_hold)
#    np.save('gan_test_'+str(latent_dim)+'dim_kl_factor_0.01_test_ssim.npy', ssim_hold)
#    np.save('gan_test_'+str(latent_dim)+'dim_kl_factor_0.01_test_abs_error.npy', l1_hold)
#    np.save('gan_test_'+str(latent_dim)+'dim_kl_factor_0.01_test_nrmse.npy', nrmse)
  
