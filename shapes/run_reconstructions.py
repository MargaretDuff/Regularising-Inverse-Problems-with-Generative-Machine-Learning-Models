# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 12:51:11 2020

@author: marga
"""


import vae_2d_shapes_class as vae
import ae_2d_shapes_class as ae
import wgan_2d_shapes_class as gan 

import numpy as np 
#import matplotlib.pyplot as plt

import tensorflow as tf
#%%
test_images=np.load('../../datasets/2d_shapes/2d_shapes_test_images2_none.npy')
#test_rect=np.load('../../datasets/2d_shapes/2d_shapes_test_images2_rect.npy')
#test_circle=np.load('../../datasets/2d_shapes/2d_shapes_test_images2_circle.npy')











for num ,latent_dim in enumerate([10]):

    tf.reset_default_graph()
    sess1 = tf.Session()
    vae_none=vae.shapes_VAE(latent_dim,'../../2d_shapes_vae/checkpoints_no_sigmoid/checkpoints_none_'+str(latent_dim)+'_kl_factor_0.01/2d_shapes_VAE-29800',sess1)



    vae_test=np.load('vae_test_'+str(latent_dim)+'dim_kl_factor_0.01_test_nrmse.npy')

    vae_test_enc=np.load('vae_test2_none_train_ns_'+str(latent_dim)+'dim_kl_factor_0.01_noneEncodings.npy')



    vae_test_percentiles=np.zeros(11)
    for i in range(11):
        vae_test_percentiles[i]=int(abs(vae_test-np.percentile(vae_test,i*10,interpolation='nearest')).argmin())
    #np.save('vae_test_percentiles.npy', vae_test_percentiles)

    hold=np.zeros((11,1))
    for j in range(11):
        np.save('vae_test_'+str(latent_dim)+'dim_kl_factor_0.01_nrmse_reconstructions_'+str(j)+'.npy',vae_none.generate(vae_test_enc[int(vae_test_percentiles[j])])[0,:,:])
        np.save('vae_test_'+str(latent_dim)+'dim_kl_factor_0.01_nrmse_original_'+str(j)+'.npy',test_images[int(vae_test_percentiles[j])])
        hold[j]=vae_test[int(vae_test_percentiles[j])]
    np.save('vae_test_'+str(latent_dim)+'dim_kl_factor_0.01_nrmse_percentile_losses.npy', hold)

for num ,latent_dim in enumerate([10]):

    tf.reset_default_graph()
    sess1 = tf.Session()
    ae_none=ae.shapes_AE(latent_dim,'../../2d_shapes_ae/checkpoints_no_sigmoid/checkpoints_none_'+str(latent_dim)+'/2d_shapes_AE-29800',sess1)
    



    ae_test=np.load('ae_test_'+str(latent_dim)+'dim_kl_factor_0.01_test_nrmse.npy')

    ae_test_enc=np.load('ae_test2_none_train_ns_'+str(latent_dim)+'dim_noneEncodings.npy')



    ae_test_percentiles=np.zeros(11)
    for i in range(11):
        ae_test_percentiles[i]=int(abs(ae_test-np.percentile(ae_test,i*10,interpolation='nearest')).argmin())
    #np.save('vae_test_percentiles.npy', vae_test_percentiles)

    hold=np.zeros((11,1))
    for j in range(11):
        np.save('ae_test_'+str(latent_dim)+'dim_kl_factor_0.01_nrmse_reconstructions_'+str(j)+'.npy',ae_none.generate(ae_test_enc[int(ae_test_percentiles[j])])[0,:,:])
        np.save('ae_test_'+str(latent_dim)+'dim_kl_factor_0.01_nrmse_original_'+str(j)+'.npy',test_images[int(ae_test_percentiles[j])])
        hold[j]=ae_test[int(ae_test_percentiles[j])]
    np.save('ae_test_'+str(latent_dim)+'dim_kl_factor_0.01_nrmse_percentile_losses.npy', hold)

for num ,latent_dim in enumerate([10]):

    tf.reset_default_graph()
    sess1 = tf.Session()
    gan_none=gan.shapesGAN(latent_dim,'../../2d_shapes_wgan/checkpoints_ns_'+str(latent_dim)+'_none/2d_shapes_wgan_'+str(latent_dim)+'dim-29800',sess1)



    gan_test=np.load('gan_test_'+str(latent_dim)+'dim_kl_factor_0.01_test_nrmse.npy')

    gan_test_enc=np.load('gan_test2_none_train_ns_'+str(latent_dim)+'dim_noneEncodings.npy')



    gan_test_percentiles=np.zeros(11)
    for i in range(11):
        gan_test_percentiles[i]=int(abs(gan_test-np.percentile(gan_test,i*10,interpolation='nearest')).argmin())
    #np.save('vae_test_percentiles.npy', vae_test_percentiles)

    hold=np.zeros((11,1))
    for j in range(11):
        np.save('gan_test_'+str(latent_dim)+'dim_kl_factor_0.01_nrmse_reconstructions_'+str(j)+'.npy',gan_none.generate(gan_test_enc[int(gan_test_percentiles[j])])[0,:,:])
        np.save('gan_test_'+str(latent_dim)+'dim_kl_factor_0.01_nrmse_original_'+str(j)+'.npy',test_images[int(gan_test_percentiles[j])])
        hold[j]=gan_test[int(gan_test_percentiles[j])]
    np.save('gan_test_'+str(latent_dim)+'dim_kl_factor_0.01_nrmse_percentile_losses.npy', hold)




#for num ,latent_dim in enumerate([5,6,7,8,9,10,11,12,13,14,16,18,20,25]):
 
#     tf.reset_default_graph()
#     sess1 = tf.Session()
#     vae_none=vae.shapes_VAE(latent_dim,'../../2d_shapes_vae/checkpoints_no_sigmoid/checkpoints_none_'+str(latent_dim)+'_kl_factor_0.01/2d_shapes_VAE-29800',sess1)
  
    
    
#     vae_train_none_test_none=np.load('vae_test2_none_train_ns_'+str(latent_dim)+'dim_kl_factor_0.01_noneInOutLoss.npy')
#     vae_train_none_test_circle=np.load('vae_test2_circle_train_ns_'+str(latent_dim)+'dim_noneInOutLoss.npy')
#     vae_train_none_test_rect=np.load('vae_test2_rect_train_ns_'+str(latent_dim)+'dim_noneInOutLoss.npy')
    
#     vae_train_none_test_none_enc=np.load('vae_test2_none_train_ns_'+str(latent_dim)+'dim_noneEncodings.npy')
#     vae_train_none_test_circle_enc=np.load('vae_test2_circle_train_ns_'+str(latent_dim)+'dim_noneEncodings.npy')
#     vae_train_none_test_rect_enc=np.load('vae_test2_rect_train_ns_'+str(latent_dim)+'dim_noneEncodings.npy')
    
#     vae_train_none_test_circle_mask=np.load('vae_test2_circle_train_ns_'+str(latent_dim)+'dim_noneMaskSpotBack.npy')
#     vae_train_none_test_circle_mask=(np.abs(vae_train_none_test_circle_mask[:,0]-1)+np.abs(vae_train_none_test_circle_mask[:,1]-0.5))/2
#     vae_train_none_test_rect_mask=np.load('vae_test2_rect_train_ns_'+str(latent_dim)+'dim_noneMaskSpotBack.npy')
#     vae_train_none_test_rect_mask=(np.abs(vae_train_none_test_rect_mask[:,0]-1)+np.abs(vae_train_none_test_rect_mask[:,1]-0.5))/2
#     #%%
    
    
#     vae_train_none_test_rect_percentiles=np.zeros(11)
#     for i in range(11):
#         vae_train_none_test_rect_percentiles[i]=int(abs(vae_train_none_test_rect-np.percentile(vae_train_none_test_rect,i*10,interpolation='nearest')).argmin())
#     #np.save('vae_train_none_test_rect_percentiles.npy', vae_train_none_test_rect_percentiles)
    
#     vae_train_none_test_circle_percentiles=np.zeros(11)
#     for i in range(11):
#         vae_train_none_test_circle_percentiles[i]=int(abs(vae_train_none_test_circle-np.percentile(vae_train_none_test_circle,i*10,interpolation='nearest')).argmin())
#     #np.save('vae_train_none_test_circle_percentiles.npy', vae_train_none_test_circle_percentiles)
    
#     vae_train_none_test_none_percentiles=np.zeros(11)
#     for i in range(11):
#         vae_train_none_test_none_percentiles[i]=int(abs(vae_train_none_test_none-np.percentile(vae_train_none_test_none,i*10,interpolation='nearest')).argmin())
#     np.save('vae_train_none_test_none_percentiles.npy', vae_train_none_test_none_percentiles)
    
#     vae_train_none_test_circle_percentiles_mask=np.zeros(11)
#     for i in range(11):
#         vae_train_none_test_circle_percentiles_mask[i]=int(abs(vae_train_none_test_circle_mask-np.percentile(vae_train_none_test_circle_mask,i*10,interpolation='nearest')).argmin())
    
    
#     vae_train_none_test_rect_percentiles_mask=np.zeros(11)
#     for i in range(11):
#         vae_train_none_test_rect_percentiles_mask[i]=int(abs(vae_train_none_test_rect_mask-np.percentile(vae_train_none_test_rect_mask,i*10,interpolation='nearest')).argmin())
    
    

hold=np.zeros((11,1))    

#     for j in range(11):
#         np.save('vae_train_none_'+str(latent_dim)+'dim_test_none_reconstructions_'+str(j)+'.npy',vae_none.generate(vae_train_none_test_none_enc[int(vae_train_none_test_none_percentiles[j])])[0,:,:])
#         np.save('vae_train_none_'+str(latent_dim)+'dim_test_none_original_'+str(j)+'.npy',test_none[int(vae_train_none_test_none_percentiles[j])])
#         print(np.shape(vae_train_none_test_none))
#         print(np.shape(vae_train_none_test_none_percentiles))
#         hold[j]=vae_train_none_test_none[int(vae_train_none_test_none_percentiles[j])]
#     np.save('vae_train_none_'+str(latent_dim)+'dim_test_none_percentile_losses.npy', hold)





#%%

#for num ,latent_dim in enumerate([5,6,7,8,9,10,11,12,13,14]):
 
#    tf.reset_default_graph()
#    sess1 = tf.Session()
#    ae_none=ae.shapes_AE(latent_dim,'../../2d_shapes_ae/checkpoints_no_sigmoid/checkpoints_none_'+str(latent_dim)+'/2d_shapes_AE-29800',sess1)
  
    
    
#    ae_train_none_test_none=np.load('ae_test2_none_train_ns_'+str(latent_dim)+'dim_noneInOutLoss.npy')
#    ae_train_none_test_circle=np.load('ae_test2_circle_train_ns_'+str(latent_dim)+'dim_noneInOutLoss.npy')
#    ae_train_none_test_rect=np.load('ae_test2_rect_train_ns_'+str(latent_dim)+'dim_noneInOutLoss.npy')
#    
#    ae_train_none_test_none_enc=np.load('ae_test2_none_train_ns_'+str(latent_dim)+'dim_noneEncodings.npy')
#    ae_train_none_test_circle_enc=np.load('ae_test2_circle_train_ns_'+str(latent_dim)+'dim_noneEncodings.npy')
#    ae_train_none_test_rect_enc=np.load('ae_test2_rect_train_ns_'+str(latent_dim)+'dim_noneEncodings.npy')
    
#    ae_train_none_test_circle_mask=np.load('ae_test2_circle_train_ns_'+str(latent_dim)+'dim_noneMaskSpotBack.npy')
#    ae_train_none_test_circle_mask=(np.abs(ae_train_none_test_circle_mask[:,0]-1)+np.abs(ae_train_none_test_circle_mask[:,1]-0.5))/2
#    ae_train_none_test_rect_mask=np.load('ae_test2_rect_train_ns_'+str(latent_dim)+'dim_noneMaskSpotBack.npy')
#    ae_train_none_test_rect_mask=(np.abs(ae_train_none_test_rect_mask[:,0]-1)+np.abs(ae_train_none_test_rect_mask[:,1]-0.5))/2
    #%%
    
    
#    ae_train_none_test_rect_percentiles=np.zeros(11)
#    for i in range(11):
#        ae_train_none_test_rect_percentiles[i]=int(abs(ae_train_none_test_rect-np.percentile(ae_train_none_test_rect,i*10,interpolation='nearest')).argmin())
    #np.save('ae_train_none_test_rect_percentiles.npy', ae_train_none_test_rect_percentiles)
    
#    ae_train_none_test_circle_percentiles=np.zeros(11)
#    for i in range(11):
#        ae_train_none_test_circle_percentiles[i]=int(abs(ae_train_none_test_circle-np.percentile(ae_train_none_test_circle,i*10,interpolation='nearest')).argmin())
    #np.save('ae_train_none_test_circle_percentiles.npy', ae_train_none_test_circle_percentiles)
    
#    ae_train_none_test_none_percentiles=np.zeros(11)
#    for i in range(11):
#        ae_train_none_test_none_percentiles[i]=int(abs(ae_train_none_test_none-np.percentile(ae_train_none_test_none,i*10,interpolation='nearest')).argmin())
    #np.save('ae_train_none_test_none_percentiles.npy', ae_train_none_test_none_percentiles)
    
#    ae_train_none_test_circle_percentiles_mask=np.zeros(11)
#    for i in range(11):
#        ae_train_none_test_circle_percentiles_mask[i]=int(abs(ae_train_none_test_circle_mask-np.percentile(ae_train_none_test_circle_mask,i*10,interpolation='nearest')).argmin())
    
    
#    ae_train_none_test_rect_percentiles_mask=np.zeros(11)
#    for i in range(11):
#        ae_train_none_test_rect_percentiles_mask[i]=int(abs(ae_train_none_test_rect_mask-np.percentile(ae_train_none_test_rect_mask,i*10,interpolation='nearest')).argmin())
    

#    hold=np.zeros((11,1))    

#    for j in range(11):
#        np.save('ae_train_none_'+str(latent_dim)+'dim_test_none_reconstructions_'+str(j)+'.npy',ae_none.generate(ae_train_none_test_none_enc[int(ae_train_none_test_none_percentiles[j])])[0,:,:])
#        np.save('ae_train_none_'+str(latent_dim)+'dim_test_none_original_'+str(j)+'.npy',test_none[int(ae_train_none_test_none_percentiles[j])])
#        print(np.shape(ae_train_none_test_none))
#        print(np.shape(ae_train_none_test_none_percentiles))
#        hold[j]=ae_train_none_test_none[int(ae_train_none_test_none_percentiles[j])]
#    np.save('ae_train_none_'+str(latent_dim)+'dim_test_none_percentile_losses.npy', hold)






 #   for j in range(11):
#        np.save('ae_train_none_'+str(latent_dim)+'dim_test_circle_reconstructions_'+str(j)+'.npy',ae_none.generate(ae_train_none_test_circle_enc[int(ae_train_none_test_circle_percentiles[j])])[0,:,:])
#        np.save('ae_train_none_'+str(latent_dim)+'dim_test_circle_original_'+str(j)+'.npy',test_circle[int(ae_train_none_test_circle_percentiles[j])])
#        hold[j]=ae_train_none_test_circle[int(ae_train_none_test_circle_percentiles[j])]
#    np.save('ae_train_none_'+str(latent_dim)+'dim_test_circle_percentile_losses.npy', hold)



#    for j in range(11):
#        np.save('ae_train_none_'+str(latent_dim)+'dim_test_rect_reconstructions_'+str(j)+'.npy',ae_none.generate(ae_train_none_test_rect_enc[int(ae_train_none_test_rect_percentiles[j])])[0,:,:])
#        np.save('ae_train_none_'+str(latent_dim)+'dim_test_rect_original_'+str(j)+'.npy',test_rect[int(ae_train_none_test_rect_percentiles[j])])
#        hold[j]=ae_train_none_test_rect[int(ae_train_none_test_rect_percentiles[j])]
#    np.save('ae_train_none_'+str(latent_dim)+'dim_test_rect_percentile_losses.npy', hold)



#    for j in range(11):
#        np.save('ae_train_none_'+str(latent_dim)+'dim_test_circle_reconstructions_mask_'+str(j)+'.npy',ae_none.generate(ae_train_none_test_circle_enc[int(ae_train_none_test_circle_percentiles_mask[j])])[0,:,:])
#        np.save('ae_train_none_'+str(latent_dim)+'dim_test_circle_original_mask_'+str(j)+'.npy',test_circle[int(ae_train_none_test_circle_percentiles_mask[j])])
#        hold[j]=ae_train_none_test_circle_mask[int(ae_train_none_test_circle_percentiles_mask[j])]

#    np.save('ae_train_none_'+str(latent_dim)+'dim_test_circle_percentile_losses_mask.npy', hold)



#    for j in range(11):
#        np.save('ae_train_none_'+str(latent_dim)+'dim_test_rect_reconstructions_mask_'+str(j)+'.npy',ae_none.generate(ae_train_none_test_rect_enc[int(ae_train_none_test_rect_percentiles_mask[j])])[0,:,:])
#        np.save('ae_train_none_'+str(latent_dim)+'dim_test_rect_original_mask_'+str(j)+'.npy',test_rect[int(ae_train_none_test_rect_percentiles_mask[j])])
#        hold[j]=ae_train_none_test_rect_mask[int(ae_train_none_test_rect_percentiles_mask[j])]

#    np.save('ae_train_none_'+str(latent_dim)+'dim_test_rect_percentile_losses_mask.npy', hold)



#%%

#for num ,latent_dim in enumerate([10,8]):
 
#    tf.reset_default_graph()
#    sess1 = tf.Session()
#    gan_none= gan.shapesGAN(latent_dim,'../../2d_shapes_wgan/checkpoints_ns_'+str(latent_dim)+'_none/2d_shapes_wgan_'+str(latent_dim)+'dim-29800',sess1)
  
    
    
#    gan_train_none_test_none=np.load('gan_test2_none_train_ns_'+str(latent_dim)+'dim_noneInOutLoss.npy')
#    gan_train_none_test_circle=np.load('gan_test2_circle_train_ns_'+str(latent_dim)+'dim_noneInOutLoss.npy')
#    gan_train_none_test_rect=np.load('gan_test2_rect_train_ns_'+str(latent_dim)+'dim_noneInOutLoss.npy')
    
#    gan_train_none_test_none_enc=np.load('gan_test2_none_train_ns_'+str(latent_dim)+'dim_noneEncodings.npy')
#    gan_train_none_test_circle_enc=np.load('gan_test2_circle_train_ns_'+str(latent_dim)+'dim_noneEncodings.npy')
#    gan_train_none_test_rect_enc=np.load('gan_test2_rect_train_ns_'+str(latent_dim)+'dim_noneEncodings.npy')
    
#    gan_train_none_test_circle_mask=np.load('gan_test2_circle_train_ns_'+str(latent_dim)+'dim_noneMaskSpotBack.npy')
#    gan_train_none_test_circle_mask=(np.abs(gan_train_none_test_circle_mask[:,0]-1)+np.abs(gan_train_none_test_circle_mask[:,1]-0.5))/2
#    gan_train_none_test_rect_mask=np.load('gan_test2_rect_train_ns_'+str(latent_dim)+'dim_noneMaskSpotBack.npy')
#    gan_train_none_test_rect_mask=(np.abs(gan_train_none_test_rect_mask[:,0]-1)+np.abs(gan_train_none_test_rect_mask[:,1]-0.5))/2
    #%%
    
    
 #   gan_train_none_test_rect_percentiles=np.zeros(11)
 #   for i in range(11):
 #       gan_train_none_test_rect_percentiles[i]=int(abs(gan_train_none_test_rect-np.percentile(gan_train_none_test_rect,i*10,interpolation='nearest')).argmin())
    #np.save('gan_train_none_test_rect_percentiles.npy', gan_train_none_test_rect_percentiles)
    
 #   gan_train_none_test_circle_percentiles=np.zeros(11)
 #   for i in range(11):
 #       gan_train_none_test_circle_percentiles[i]=int(abs(gan_train_none_test_circle-np.percentile(gan_train_none_test_circle,i*10,interpolation='nearest')).argmin())
    #np.save('gan_train_none_test_circle_percentiles.npy', gan_train_none_test_circle_percentiles)
    
 #   gan_train_none_test_none_percentiles=np.zeros(11)
 #   for i in range(11):
 #       gan_train_none_test_none_percentiles[i]=int(abs(gan_train_none_test_none-np.percentile(gan_train_none_test_none,i*10,interpolation='nearest')).argmin())
    #np.save('gan_train_none_test_none_percentiles.npy', gan_train_none_test_none_percentiles)
    
 #   gan_train_none_test_circle_percentiles_mask=np.zeros(11)
 #   for i in range(11):
 #       gan_train_none_test_circle_percentiles_mask[i]=int(abs(gan_train_none_test_circle_mask-np.percentile(gan_train_none_test_circle_mask,i*10,interpolation='nearest')).argmin())
    
    
 #   gan_train_none_test_rect_percentiles_mask=np.zeros(11)
 #   for i in range(11):
 #       gan_train_none_test_rect_percentiles_mask[i]=int(abs(gan_train_none_test_rect_mask-np.percentile(gan_train_none_test_rect_mask,i*10,interpolation='nearest')).argmin())
    

 #   hold=np.zeros((11,1))    

#    for j in range(11):
#        np.save('gan_train_none_'+str(latent_dim)+'dim_test_none_reconstructions_'+str(j)+'.npy',gan_none.generate(gan_train_none_test_none_enc[int(gan_train_none_test_none_percentiles[j])])[0,:,:])
#        np.save('gan_train_none_'+str(latent_dim)+'dim_test_none_original_'+str(j)+'.npy',test_none[int(gan_train_none_test_none_percentiles[j])])
#        print(np.shape(gan_train_none_test_none))
#        print(np.shape(gan_train_none_test_none_percentiles))
#        hold[j]=gan_train_none_test_none[int(gan_train_none_test_none_percentiles[j])]
#    np.save('gan_train_none_'+str(latent_dim)+'dim_test_none_percentile_losses.npy', hold)






#    for j in range(11):
#        np.save('gan_train_none_'+str(latent_dim)+'dim_test_circle_reconstructions_'+str(j)+'.npy',gan_none.generate(gan_train_none_test_circle_enc[int(gan_train_none_test_circle_percentiles[j])])[0,:,:])
#        np.save('gan_train_none_'+str(latent_dim)+'dim_test_circle_original_'+str(j)+'.npy',test_circle[int(gan_train_none_test_circle_percentiles[j])])
#        hold[j]=gan_train_none_test_circle[int(gan_train_none_test_circle_percentiles[j])]
#    np.save('gan_train_none_'+str(latent_dim)+'dim_test_circle_percentile_losses.npy', hold)



#    for j in range(11):
#        np.save('gan_train_none_'+str(latent_dim)+'dim_test_rect_reconstructions_'+str(j)+'.npy',gan_none.generate(gan_train_none_test_rect_enc[int(gan_train_none_test_rect_percentiles[j])])[0,:,:])
#        np.save('gan_train_none_'+str(latent_dim)+'dim_test_rect_original_'+str(j)+'.npy',test_rect[int(gan_train_none_test_rect_percentiles[j])])
#        hold[j]=gan_train_none_test_rect[int(gan_train_none_test_rect_percentiles[j])]
#    np.save('gan_train_none_'+str(latent_dim)+'dim_test_rect_percentile_losses.npy', hold)



#    for j in range(11):
#        np.save('gan_train_none_'+str(latent_dim)+'dim_test_circle_reconstructions_mask_'+str(j)+'.npy',gan_none.generate(gan_train_none_test_circle_enc[int(gan_train_none_test_circle_percentiles_mask[j])])[0,:,:])
#        np.save('gan_train_none_'+str(latent_dim)+'dim_test_circle_original_mask_'+str(j)+'.npy',test_circle[int(gan_train_none_test_circle_percentiles_mask[j])])
#        hold[j]=gan_train_none_test_circle_mask[int(gan_train_none_test_circle_percentiles_mask[j])]

#    np.save('gan_train_none_'+str(latent_dim)+'dim_test_circle_percentile_losses_mask.npy', hold)



#    for j in range(11):
#        np.save('gan_train_none_'+str(latent_dim)+'dim_test_rect_reconstructions_mask_'+str(j)+'.npy',gan_none.generate(gan_train_none_test_rect_enc[int(gan_train_none_test_rect_percentiles_mask[j])])[0,:,:])
#        np.save('gan_train_none_'+str(latent_dim)+'dim_test_rect_original_mask_'+str(j)+'.npy',test_rect[int(gan_train_none_test_rect_percentiles_mask[j])])
#        hold[j]=gan_train_none_test_rect_mask[int(gan_train_none_test_rect_percentiles_mask[j])]

#    np.save('gan_train_none_'+str(latent_dim)+'dim_test_rect_percentile_losses_mask.npy', hold)

