
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
    vae_none=vae.mnistVAE(latent_dim,'checkpoints/vae/checkpoints_no_sigmoid_'+str(latent_dim)+'_kl_factor_0.05/checkpoints'+str(latent_dim)+'-29800')
    for i in range(10):
        img=vae_none.generate(np.random.normal(0, 50, latent_dim))
        np.save('mnist_vae_'+str(latent_dim)+'dim_kl_factor_0.05_far_from_distribution_'+str(i)+'.npy', img)
    


for num ,latent_dim in enumerate([8]):

    tf.reset_default_graph()
    sess1 = tf.Session()
    ae_none=ae.mnistAE(latent_dim,'checkpoints/ae/checkpoints'+str(latent_dim)+'/mnist_AE_'+str(latent_dim)+'-29800')
    for i in range(10):
        img=ae_none.generate(np.random.normal(0, 50, latent_dim))
        np.save('mnist_ae_'+str(latent_dim)+'dim_far_from_distribution_'+str(i)+'.npy', img)



for num ,latent_dim in enumerate([8]):

    tf.reset_default_graph()
    sess1 = tf.Session()
    gan_none=gan.mnistGAN(latent_dim,'checkpoints/wgan/checkpoints_gan_'+str(latent_dim)+'/mnist_GAN_'+str(latent_dim)+'-19900')
    for i in range(10):
        img=gan_none.generate(np.random.normal(0, 50, latent_dim))
        np.save('mnist_gan_'+str(latent_dim)+'dim_far_from_distribution_'+str(i)+'.npy', img)



