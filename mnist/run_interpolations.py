import numpy as np

import mnist_vae_ns_test as vae
import mnist_ae_test as ae
import mnist_gan_test as gan
import tensorflow as tf
test_images=np.load('mnist_test_images.npy')


for num ,latent_dim in enumerate([5,6,7,8,9,10,11,12,13,14,16,18,20,25,30,40,50,75,100]):

    tf.reset_default_graph()
    sess1 = tf.Session()
    vae_none=vae.mnistVAE(latent_dim,'../mnist_vae/checkpoints_no_sigmoid_'+str(latent_dim)+'_kl_factor_0.05/checkpoints'+str(latent_dim)+'-29800')

    vae_none.interpolation( vae_none.encode(test_images[105]),vae_none.encode(test_images[205]),vae_none.encode(test_images[305]), intervals=10, save=True, save_name='mnsit_vae_'+str(latent_dim)+'dim_interpolations_kl_005')


#for num ,latent_dim in enumerate([5,6,7,8,9,10,11,12,13,14,16,18,20,25,30,40,50,75,100]):#
#
#    tf.reset_default_graph()
#    sess1 = tf.Session()
#    ae_none=ae.mnistAE(latent_dim,'../ae_mnist/checkpoints_ns/checkpoints'+str(latent_dim)+'/mnist_AE_'+str(latent_dim)+'-29800')


#    ae_none.interpolation( ae_none.encode(test_images[105]),ae_none.encode(test_images[205]),ae_none.encode(test_images[305]), intervals=10, save=True, save_name='mnsit_ae_'+str(latent_dim)+'dim_interpolations2')



#for num ,latent_dim in enumerate([5,6,7,8,9,10,11,12,13,14,16,18,20,25,30,40,50,75,100]):

#    tf.reset_default_graph()
#    sess1 = tf.Session()
#    gan_none=gan.mnistGAN(latent_dim,'../mnist_gan/checkpoints_ns/checkpoints_gan_'+str(latent_dim)+'/mnist_GAN_'+str(latent_dim)+'-19900')

#    gan_none.interpolation( gan_none.encode(test_images[105]),gan_none.encode(test_images[205]),gan_none.encode(test_images[305]), intervals=10, save=True, save_name='mnsit_gan_'+str(latent_dim)+'dim_interpolations2')
