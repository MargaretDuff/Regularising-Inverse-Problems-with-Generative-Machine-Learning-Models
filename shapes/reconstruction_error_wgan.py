# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 15:18:09 2020

@author: marga
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 10:34:19 2020

@author: marga
"""
import numpy as np
from scipy import stats

#import matplotlib.pyplot as plt


import generative_model as test_class
#import ae_2d_shapes_class as ae
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
    
    test_images_none=np.load('../../datasets/2d_shapes/2d_shapes_test_images2_none.npy')
    #test_images_circle=np.load('../../datasets/2d_shapes/2d_shapes_test_images2_circle.npy')
    #test_images_rect=np.load('../../datasets/2d_shapes/2d_shapes_test_images2_rect.npy')

#    rect_mask_back=np.load('../../datasets/2d_shapes/2d_shapes_test_images2_mask_background_rect.npy')
 #   rect_mask_spot=np.load('../../datasets/2d_shapes/2d_shapes_test_images2_mask_spot_rect.npy')
  #  circle_mask_spot=np.load('../../datasets/2d_shapes/2d_shapes_test_images2_mask_spot_circle.npy')
   # circle_mask_back=np.load('../../datasets/2d_shapes/2d_shapes_test_images2_mask_background_circle.npy')
    
    
 
    print('Loaded images')

import tensorflow as tf 

#for i in [13]:

 #   tf.reset_default_graph()
  #  sess_none = tf.InteractiveSession()
   # ae_test_none=ae.shapes_AE(i, '../../2d_shapes_ae/checkpoints_no_sigmoid/checkpoints_none_'+str(i)+'/2d_shapes_AE-29800', sess_none)
    #ae_test_none.latentSpaceValues(test_images_none, saveName='ae_test2_none_train_ns_'+str(i)+'dim_none')

for i in [8,10,12]:
#for i in [7,9,11,13,14,16,18,20,25,30,40]:
#for i in [30,40,50,75,100]:

    tf.reset_default_graph()
    sess_none = tf.InteractiveSession()
    gan_test_none=gan.shapesGAN(i, '../../2d_shapes_wgan/checkpoints_ns_'+str(i)+'_none/2d_shapes_wgan_'+str(i)+'dim-29800', sess_none)
#    gan_test_none.reconstructionInOutandMask(test_images_circle,circle_mask_back, circle_mask_spot,saveName='gan_test2_circle_train_ns_'+str(i)+'dim_none')
    gan_test_none.latentSpaceValues(test_images_none[:1000], saveName='gan_test2_none_train_ns_'+str(i)+'dim_none_diff_loss_more')
#    gan_test_none.reconstructionInOutandMask(test_images_rect,rect_mask_back,rect_mask_spot, saveName='gan_test2_rect_train_ns_'+str(i)+'dim_none')


