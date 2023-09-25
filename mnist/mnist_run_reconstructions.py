# *- coding: utf-8 -*-
"""
Created on Tue Jul 14 12:51:11 2020

@author: marga
"""


import mnist_vae_ns_test as vae
import mnist_ae_test as ae
import mnist_gan_test as gan 
import numpy as np 
#import matplotlib.pyplot as plt

import tensorflow as tf
#%%
test_images=np.load('mnist_test_images.npy')


for num ,latent_dim in enumerate([8]):
 
    tf.reset_default_graph()
    sess1 = tf.Session()
    vae_none=vae.mnistVAE(latent_dim,'../mnist_vae/checkpoints_no_sigmoid_'+str(latent_dim)+'_kl_factor_0.05/checkpoints'+str(latent_dim)+'-29800')
  
    
    
    vae_test=np.load('vae_test_'+str(latent_dim)+'dim_kl_factor_0.05_test_nrmse.npy')
   
    vae_test_enc=np.load('vae_ns_train_images_'+str(latent_dim)+'dim_kl_factor_0.05_Encodings.npy')
    
    
    
    vae_test_percentiles=np.zeros(11)
    for i in range(11):
        vae_test_percentiles[i]=int(abs(vae_test-np.percentile(vae_test,i*10,interpolation='nearest')).argmin())
    np.save('vae_test_percentiles.npy', vae_test_percentiles)
    
    hold=np.zeros((11,1))
    for j in range(11):
        np.save('vae_test_'+str(latent_dim)+'dim_kl_factor_0.05_nrmse_reconstructions_'+str(j)+'.npy',vae_none.generate(vae_test_enc[int(vae_test_percentiles[j])])[0,:,:])
        np.save('vae_test_'+str(latent_dim)+'dim_kl_factor_0.05_nrmse_original_'+str(j)+'.npy',test_images[int(vae_test_percentiles[j])])
        hold[j]=vae_test[int(vae_test_percentiles[j])]
    np.save('vae_test_'+str(latent_dim)+'dim_kl_factor_0.05_rmse_percentile_losses.npy', hold)

 
for num ,latent_dim in enumerate([5,6,8,9,10,13,16]):

# 
    tf.reset_default_graph()
    sess1 = tf.Session()
    ae_none=ae.mnistAE(latent_dim,'../ae_mnist/checkpoints_ns/checkpoints'+str(latent_dim)+'/mnist_AE_'+str(latent_dim)+'-29800')
#  
    
    
    ae_test=np.load('ae_test_'+str(latent_dim)+'dim_kl_factor_0.05_test_nrmse.npy')
   
    ae_test_enc=np.load('ae_ns_test_images_'+str(latent_dim)+'dimEncodings.npy')
#    
    
    
    ae_test_percentiles=np.zeros(11)
    for i in range(11):
        ae_test_percentiles[i]=int(abs(ae_test-np.percentile(ae_test,i*10,interpolation='nearest')).argmin())
   #np.save('ae_test_percentiles.npy', ae_test_percentiles)
    
 
    for j in range(11):
        np.save('ae_test_'+str(latent_dim)+'dim_nrmse_reconstructions_'+str(j)+'.npy',ae_none.generate(ae_test_enc[int(ae_test_percentiles[j])])[0,:,:])
        np.save('ae_test_'+str(latent_dim)+'dim_nrmse_original_'+str(j)+'.npy',test_images[int(ae_test_percentiles[j])])
        hold[j]=ae_test[int(ae_test_percentiles[j])]
    np.save('ae_test_'+str(latent_dim)+'dim_nrmse_percentile_losses.npy', hold)

for num ,latent_dim in enumerate([8]):
 
    tf.reset_default_graph()
    sess1 = tf.Session()
    gan_none=gan.mnistGAN(latent_dim,'../mnist_gan/checkpoints_ns/checkpoints_gan_'+str(latent_dim)+'/mnist_GAN_'+str(latent_dim)+'-19900')
  
    
    
    gan_test=np.load('gan_test_'+str(latent_dim)+'dim_kl_factor_0.05_test_nrmse.npy')
   
    gan_test_enc=np.load('gan_ns_test_images_'+str(latent_dim)+'dimEncodings.npy')
    
    
    
    gan_test_percentiles=np.zeros(11)
    for i in range(11):
        gan_test_percentiles[i]=int(abs(gan_test-np.percentile(gan_test,i*10,interpolation='nearest')).argmin())
    #np.save('gan_test_percentiles.npy', gan_test_percentiles)
    
 
    for j in range(11):
        np.save('gan_test_'+str(latent_dim)+'dim_nrmse_reconstructions_'+str(j)+'.npy',gan_none.generate(gan_test_enc[int(gan_test_percentiles[j])])[0,:,:])
        np.save('gan_test_'+str(latent_dim)+'dim_nrmse_original_'+str(j)+'.npy',test_images[int(gan_test_percentiles[j])])
        hold[j]=gan_test[int(gan_test_percentiles[j])]
    np.save('gan_test_'+str(latent_dim)+'dim_nrmse_percentile_losses.npy', hold)


