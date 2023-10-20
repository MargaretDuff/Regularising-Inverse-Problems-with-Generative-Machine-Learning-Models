# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 10:34:19 2020

@author: marga
"""
import numpy as np
from scipy import stats
import tensorflow as tf
# import matplotlib.pyplot as plt


import generative_model as test_class
import vae_2d_shapes_class_odl as vae
import ae_2d_shapes_class_odl as ae
import gan_2d_shapes_class_odl as gan

import sys
import getopt

try:
    opts, args = getopt.getopt(sys.argv[1:], "h")
except getopt.GetoptError:
    print('in_out_loss.py ')
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print('in_out_loss.py')
        sys.exit()


if __name__ == '__main__':

    test_images_none = np.load(
        '../../datasets/2d_shapes/2d_shapes_test_images_none.npy')
    print('Loaded images')

for i in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 25, 30, 40, 50, 75, 100]:

    tf.reset_default_graph()
    sess_none = tf.InteractiveSession()
    vae_test_none = vae.shapes_VAE(
        i, '../../2d_shapes_vae/checkpoints_no_sigmoid/checkpoints_none_'+str(i)+'/2d_shapes_VAE-29800', sess_none)
    vae_test_none.latentSpaceValues(
        test_images_none, saveName='vae_test_none_train_ns_'+str(i)+'dim_none')

    tf.reset_default_graph()
    sess_none = tf.InteractiveSession()
    ae_test_none = ae.shapes_AE(
        i, '../../2d_shapes_ae/checkpoints_no_sigmoid/checkpoints_none_'+str(i)+'/2d_shapes_VAE-29800', sess_none)
    ae_test_none.latentSpaceValues(
        test_images_none, saveName='ae_test_none_train_ns_'+str(i)+'dim_none')

    tf.reset_default_graph()
    sess_none = tf.InteractiveSession()
    gan_test_none = gan.shapesGAN(
        i, '../../2d_shapes_wgan/checkpoints_ns_10_none/2d_shapes_wgan_10dim-29800', sess_none)
    gan_test_none.latentSpaceValues(
        test_images_none, saveName='test_none_train_gan_ns_'+str(i)+'dim_none')
