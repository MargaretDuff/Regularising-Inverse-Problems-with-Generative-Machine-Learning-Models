# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 10:34:19 2020

@author: marga
"""
import numpy as np
#from scipy import stats
import tensorflow as tf
import matplotlib.pyplot as plt


import generative_model as test_class
#import vae_2d_shapes_class as vae
import wgan_2d_shapes_class as gan

import sys, getopt

try:
    opts, args = getopt.getopt(sys.argv[1:],"h")
except getopt.GetoptError:
    print('in_out_loss.py ')
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print('in_out_loss.py')
        sys.exit()






if __name__=='__main__':
    
    test_images_none=np.load('../../datasets/2d_shapes/2d_shapes_test_images_none.npy')
#    test_images_circle=np.load('../../datasets/2d_shapes/2d_shapes_test_images_circle.npy')
 #   test_images_rect=np.load('../../datasets/2d_shapes/2d_shapes_test_images_rect.npy')
    print('Loaded iamges')

tf.reset_default_graph()
sess_none = tf.InteractiveSession()

gan_test_10_none=gan.shapesGAN(10, '../../2d_shapes_wgan/checkpoints_ns_10_none/2d_shapes_wgan_10dim-29800', sess_none)
#sess_circle = tf.InteractiveSession()

#gan_test_10_circle=gan.shapesGAN(10, '../../2d_shapes_wgan/checkpoints_ns_10_circle/2d_shapes_wgan_10dim-29800', sess_circle)


#sess_rect = tf.InteractiveSession()

#gan_test_10_rect=gan.shapesGAN(10, '../../2d_shapes_wgan/checkpoints_ns_10_rect/2d_shapes_wgan_10dim-29800', sess_rect)


print('Loaded models')

gan_test_10_none.latentSpaceValues(test_images_none, saveName='test_none_train_gan_ns_10dim_none')



#gan_test_10_none.latentSpaceValues(test_images_circle, saveName='test_circle_train_gan_ns_10dim_none')
#gan_test_10_none.latentSpaceValues(test_images_rect, saveName='test_rect_train_gan_ns_10dim_none')
#gan_test_10_circle.latentSpaceValues(test_images_none, saveName='test_none_train_gan_ns_10dim_circle')
#gan_test_10_circle.latentSpaceValues(test_images_circle, saveName='test_circle_train_gan_ns_10dim_circle')
#gan_test_10_circle.latentSpaceValues(test_images_rect, saveName='test_rect_train_gan_ns_10dim_circle')
#gan_test_10_rect.latentSpaceValues(test_images_none, saveName='test_none_train_gan_ns_10dim_rect')
#gan_test_10_rect.latentSpaceValues(test_images_circle, saveName='test_circle_train_gan_ns_10dim_rect')
#gan_test_10_rect.latentSpaceValues(test_images_rect, saveName='test_rect_train_gan_ns_10dim_rect')








