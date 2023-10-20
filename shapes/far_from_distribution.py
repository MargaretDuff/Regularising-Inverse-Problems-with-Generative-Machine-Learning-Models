


import vae_2d_shapes_class_odl as vae
import ae_2d_shapes_class_odl as ae
import gan_2d_shapes_class_odl as gan
import numpy as np
#import matplotlib.pyplot as plt

import tensorflow as tf



for num ,latent_dim in enumerate([5,6,7,8,9,10,11,12,13,14,16,18,20,25,30,40,50,75,100]):

    tf.reset_default_graph()
    sess1 = tf.Session()
    vae_none=vae.shapes_VAE(latent_dim,'../../2d_shapes_vae/checkpoints_no_sigmoid/checkpoints_none_'+str(latent_dim)+'_kl_factor_0.01/2d_shapes_VAE-29800',sess1)

    for i in range(10):
        img=vae_none.generate(np.random.normal(0, 50, latent_dim))
        np.save('2d_shpaes_vae_none'+str(latent_dim)+'dim_far_from_distribution_'+str(i)+'.npy', img)
    



#for num ,latent_dim in enumerate([5,6,7,8,9,10,11,12,13,14,16,18,20,25,30,40,50,75,100]):

#    tf.reset_default_graph()
#    sess1 = tf.Session()
#    ae_none=ae.shapes_AE(latent_dim,'../../2d_shapes_ae/checkpoints_no_sigmoid/checkpoints_none_'+str(latent_dim)+'/2d_shapes_AE-29800',sess1)

#    for i in range(10):
#        img=ae_none.generate(np.random.normal(0, 50, latent_dim))
#        np.save('2d_shapes_ae_none'+str(latent_dim)+'dim_far_from_distribution_'+str(i)+'.npy', img)

#for num ,latent_dim in enumerate([8,10]):

#    tf.reset_default_graph()
#    sess1 = tf.Session()

#    gan_none=gan.shapesGAN(latent_dim,'../../2d_shapes_wgan/checkpoints_ns_'+str(latent_dim)+'_none/2d_shapes_wgan_'+str(latent_dim)+'dim-29800',sess1)


#    for i in range(10):
#        img=gan_none.generate(np.random.normal(0, 50, latent_dim))
#        np.save('2d_shapes_gan_none'+str(latent_dim)+'dim_far_from_distribution_'+str(i)+'.npy', img)



