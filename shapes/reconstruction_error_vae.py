# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 10:34:19 2020

@author: marga
"""
import numpy as np
from scipy import stats

#import matplotlib.pyplot as plt


import generative_model as test_class
import vae_2d_shapes_class as vae
#import wgan_2d_shapes_class as gan

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
    
    test_images_none=np.load('../../datasets/2d_shapes/2d_shapes_test_images2_none.npy')
#    test_images_circle=np.load('../../datasets/2d_shapes/2d_shapes_test_images2_circle.npy')
#    test_images_rect=np.load('../../datasets/2d_shapes/2d_shapes_test_images2_rect.npy')

#    rect_mask_back=np.load('../../datasets/2d_shapes/2d_shapes_test_images2_mask_background_rect.npy')
#    rect_mask_spot=np.load('../../datasets/2d_shapes/2d_shapes_test_images2_mask_spot_rect.npy')
#    circle_mask_spot=np.load('../../datasets/2d_shapes/2d_shapes_test_images2_mask_spot_circle.npy')
#    circle_mask_back=np.load('../../datasets/2d_shapes/2d_shapes_test_images2_mask_background_circle.npy')
    
    
 
    print('Loaded images')
import tensorflow as tf
for i in [5,6,7,8,9,10,11,12,13,14,16,18,20,25,30,40]:

#for i in [10]:
 #   tf.reset_default_graph()
  #  sess_none = tf.InteractiveSession()
   # vae_test_none=vae.shapes_VAE(i, '../../2d_shapes_vae/checkpoints_no_sigmoid/checkpoints_none_'+str(i)+'/2d_shapes_VAE-29800', sess_none)
    # vae_test_none.latentSpaceValues(test_images_none, saveName='vae_test2_none_train_ns_'+str(i)+'dim_none')

#for i in [11,12,13,14,16,18,20,25,30,40,50,75,100]:

#for i in [30,40,50,75,100]:

    tf.reset_default_graph()
    sess_none = tf.InteractiveSession()
    vae_test_none=vae.shapes_VAE(i, '../../2d_shapes_vae/checkpoints_no_sigmoid/checkpoints_none_'+str(i)+'_kl_factor_0.05/2d_shapes_VAE-29800', sess_none)
#    vae_test_none.reconstructionInOutandMask(test_images_circle,circle_mask_back, circle_mask_spot,saveName='vae_test2_circle_train_ns_'+str(i)+'dim_none')
 #   vae_test_none.reconstructionInOutandMask(test_images_rect,rect_mask_back,rect_mask_spot, saveName='vae_test2_rect_train_ns_'+str(i)+'dim_none')
    vae_test_none.latentSpaceValues(test_images_none, saveName='vae_test2_none_train_ns_'+str(i)+'dim_kl_factor_0.05_none')









#sess_circle = tf.InteractiveSession()

#vae_test_10_circle=vae.shapes_VAE(10, '../../2d_shapes_vae/checkpoints_circle_10/2d_shapes_VAE-29800', sess_circle)
#sess_rect = tf.InteractiveSession()

#vae_test_10_rect=vae.shapes_VAE(10, '../../2d_shapes_vae/checkpoints_rect_10/2d_shapes_VAE-29800', sess_rect)




#print('Loaded models')

#vae_test_10_none.latentSpaceValues(test_images_none, saveName='vae_test_images_none_10dim_none')
#vae_test_10_circle.latentSpaceValues(test_images_circle, saveName='vae_test_images_circle_10dim_circle')
#vae_test_10_rect.latentSpaceValues(test_images_circle, saveName='vae_test_images_circle_10dim_rect')
#vae_test_10_circle.latentSpaceValues(test_images_none, saveName='vae_test_images_none_10dim_circle', restore_loss='vae_test_images_none_10dim_circleCheckpointLoss8050.npy', restore_encodings='vae_test_images_none_10dim_circleCheckpointEncodings8050.npy', start=8050)
#vae_test_10_circles.latentSpaceValues(test_images_rect, saveName='vae_test_images_rect_10dim_circles')




