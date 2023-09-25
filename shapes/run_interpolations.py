
import vae_2d_shapes_class as vae
import ae_2d_shapes_class as ae
import wgan_2d_shapes_class as gan
import numpy as np
#import matplotlib.pyplot as plt

import tensorflow as tf
#%%
test_images=np.load('../../datasets/2d_shapes/2d_shapes_test_images2_none.npy')



for num ,latent_dim in enumerate([5,6,7,8,9,10,11,12,13,14,16,18,20,25,30,40,50,75,100]):

    tf.reset_default_graph()
    sess1 = tf.Session()
    vae_none=vae.shapes_VAE(latent_dim,'../../2d_shapes_vae/checkpoints_no_sigmoid/checkpoints_none_'+str(latent_dim)+'_kl_factor_0.01/2d_shapes_VAE-29800',sess1)
    vae_none.interpolation( vae_none.encode(test_images[100]),vae_none.encode(test_images[200]),vae_none.encode(test_images[300]), intervals=10, save=True, save_name='2d_shapes_vae_none_'+str(latent_dim)+'dim_kl_factor_0.01_interpolations')






#for num ,latent_dim in enumerate([5,6,7,8,9,10,11,12,13,14,16,18,20,25,30,40,50,75,100]):

#    tf.reset_default_graph()
#    sess1 = tf.Session()
#    ae_none=ae.shapes_AE(latent_dim,'../../2d_shapes_ae/checkpoints_no_sigmoid/checkpoints_none_'+str(latent_dim)+'/2d_shapes_AE-29800',sess1)


#    ae_none.interpolation( ae_none.encode(test_images[100]),ae_none.encode(test_images[200]),ae_none.encode(test_images[300]), intervals=10, save=True, save_name='2d_shapes_ae_none_'+str(latent_dim)+'dim_interpolations')





#for num ,latent_dim in enumerate([8,10]):

#    tf.reset_default_graph()
#    sess1 = tf.Session()

#    gan_none=gan.shapesGAN(latent_dim,'../../2d_shapes_wgan/checkpoints_ns_'+str(latent_dim)+'_none/2d_shapes_wgan_'+str(latent_dim)+'dim-29800',sess1)

#    gan_none.interpolation( gan_none.encode(test_images[100]),gan_none.encode(test_images[200]),gan_none.encode(test_images[300]), intervals=10, save=True, save_name='2d_shapes_gan_none_'+str(latent_dim)+'dim_interpolations')



