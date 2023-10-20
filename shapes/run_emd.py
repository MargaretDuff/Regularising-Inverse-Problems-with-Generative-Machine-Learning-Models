import vae_2d_shapes_class_odl as vae
import ae_2d_shapes_class_odl as ae
import gan_2d_shapes_class_odl as gan 
import numpy as np

import tensorflow as tf
#%%
test_none=np.load('../../datasets/2d_shapes/2d_shapes_test_images2_none.npy')
test_rect=np.load('../../datasets/2d_shapes/2d_shapes_test_images2_rect.npy')
test_circle=np.load('../../datasets/2d_shapes/2d_shapes_test_images2_circle.npy')


emd=np.zeros((16,2))
for num ,latent_dim in enumerate([5,6,7,8,9,10,11,12,13,14,16,18,20,25,30,40]):
    emd[num,0]=latent_dim
    tf.reset_default_graph()
    sess1 = tf.Session()
    vae_none=vae.shapes_VAE(latent_dim,'../../2d_shapes_vae/checkpoints_no_sigmoid/checkpoints_none_'+str(latent_dim)+'_kl_factor_0.01/2d_shapes_VAE-29800',sess1)
    emd[num,1]=vae_none.emd_images(test_none)
    np.save('vae_2d_shapes_none_kl_factor_0.01_emd.npy',emd)


emd=np.zeros((16,2))
for num ,latent_dim in enumerate([5, 6,7,8,9,10,11,12,13,14,16,18,20,25, 30,40]):
    emd[num,0]=latent_dim
    tf.reset_default_graph()
    sess1 = tf.Session()
    ae_none=ae.shapes_AE(latent_dim,'../../2d_shapes_ae/checkpoints_no_sigmoid/checkpoints_none_'+str(latent_dim)+'/2d_shapes_AE-29800',sess1)
    emd[num,1]=ae_none.emd_images(test_none)
    np.save('ae_2d_shapes_none_emd.npy',emd)

emd=np.zeros((16,2))
for num ,latent_dim in enumerate([5,6,7,8,9,10,11,12,13,14,16,18,20,25,30,40]):
#for num ,latent_dim in enumerate([8,10]):
    emd[num,0]=latent_dim
    tf.reset_default_graph()
    sess1 = tf.Session()
    wgan_none=gan.shapesGAN(latent_dim,'../../2d_shapes_wgan/checkpoints_ns_'+str(latent_dim)+'_none/2d_shapes_wgan_'+str(latent_dim)+'dim-29800',sess1)
    emd[num,1]=wgan_none.emd_images(test_none)
    np.save('wgan_2d_shapes_none_emd.npy',emd)
