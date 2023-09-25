#-*- coding: utf-8 -*-
"""
Created on Wed Apr  8 10:34:19 2020

@author: marga
"""
import numpy as np
#from scipy import stats
import tensorflow as tf
#import matplotlib.pyplot as plt


import generative_testing_class as test_class
#import mnist_ae_test as ae
#import mnist_gan_test as gan
import mnist_vae_ns_test as vae

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
 #   train_images=np.load('mnist_train_images.npy')
    test_images=np.load('mnist_test_images.npy')
    print('Loaded images')

#
##vae_test_5=vae.mnistVAE(5 ,'../mnist_vae/checkpoints_no_sigmoid_5/checkpoints5-29800')
#vae_test_6=vae.mnistVAE(6 ,'../mnist_vae/checkpoints_no_sigmoid_6/checkpoints6-29800')
#vae_test_7=vae.mnistVAE(7 ,'../mnist_vae/checkpoints_no_sigmoid_7/checkpoints7-29800')
#vae_test_8=vae.mnistVAE(8 ,'../mnist_vae/checkpoints_no_sigmoid_8/checkpoints8-29800')
#vae_test_9=vae.mnistVAE(9 ,'../mnist_vae/checkpoints_no_sigmoid_9/checkpoints9-29800')
#vae_test_10=vae.mnistVAE(10 ,'../mnist_vae/checkpoints_no_sigmoid_10/checkpoints10-29800')
#vae_test_11=vae.mnistVAE(11 ,'../mnist_vae/checkpoints_no_sigmoid_11/checkpoints11-29800')
#vae_test_12=vae.mnistVAE(12 ,'../mnist_vae/checkpoints_no_sigmoid_12/checkpoints12-29800')
#vae_test_13=vae.mnistVAE(13 ,'../mnist_vae/checkpoints_no_sigmoid_13/checkpoints13-29800')
#vae_test_14=vae.mnistVAE(14 ,'../mnist_vae/checkpoints_no_sigmoid_14/checkpoints14-29800')
#vae_test_15=vae.mnistVAE(15 ,'../mnist_vae/checkpoints_no_sigmoid_15/checkpoints15-29800')
#vae_test_16=vae.mnistVAE(16 ,'../mnist_vae/checkpoints_no_sigmoid_16/checkpoints16-29800')
#vae_test_18=vae.mnistVAE(18 ,'../mnist_vae/checkpoints_no_sigmoid_18/checkpoints18-29800')
#vae_test_20=vae.mnistVAE(20 ,'../mnist_vae/checkpoints_no_sigmoid_20/checkpoints20-29800')
#vae_test_25=vae.mnistVAE(25 ,'../mnist_vae/checkpoints_no_sigmoid_25/checkpoints25-29800')
#vae_test_30=vae.mnistVAE(30 ,'../mnist_vae/checkpoints_no_sigmoid_30/checkpoints30-29800')
#vae_test_40=vae.mnistVAE(40 ,'../mnist_vae/checkpoints_no_sigmoid_40/checkpoints40-29800')
#vae_test_50=vae.mnistVAE(50 ,'../mnist_vae/checkpoints_no_sigmoid_50/checkpoints50-29800')
#vae_test_75=vae.mnistVAE(75 ,'../mnist_vae/checkpoints_no_sigmoid_75/checkpoints75-29800')
#vae_test_100=vae.mnistVAE(100 ,'../mnist_vae/checkpoints_no_sigmoid_100/checkpoints100-29800')
for i in [18]:
#for i in [7]:

    vae_test=vae.mnistVAE(i ,'../mnist_vae/checkpoints_no_sigmoid_'+str(i)+'_kl_factor_0.05/checkpoints'+str(i)+'-29800')
    vae_test.latentSpaceValues(test_images, saveName='vae_ns_test_images_'+str(i)+'dim_kl_factor_0.05_')
   # vae_test.latentSpaceValues(train_images[:10000], saveName='ae_ns_train_images_'+str(i)+'dim')

#Case 11, 12, 6, 14??



#vae_test_6=vae.mnistVAE(6, '../mnist_vae/checkpoints6/mnist_VAE-29800')
#vae_test_10=vae.mnistVAE(10, '../mnist_vae/checkpoints10/mnist_VAE-29800')
#vae_test_12=vae.mnistVAE(12, '../mnist_vae/checkpoints12/checkpoints12-29800')
#vae_test_14=vae.mnistVAE(14, '../mnist_vae/checkpoints14/checkpoints14-29800')

#gan_test_8=gan.mnistGAN(8, '../mnist_gan/checkpoints8/mnist_GAN_8-39900')
#gan_test_10=gan.mnistGAN(10, '../mnist_gan/checkpoints10/mnist_GAN_10-19900')
#gan_test_12=gan.mnistGAN(12, '../mnist_gan/checkpoints12/mnist_GAN_12-23600')

#print('Loaded models')
#vae_test_5.latentSpaceValues(test_images, saveName='vae_ns_test_images_5dim')
#vae_test_6.latentSpaceValues(test_images, saveName='vae_ns_test_images_6dim')
#vae_test_7.latentSpaceValues(test_images, saveName='vae_ns_test_images_7dim')
#vae_test_8.latentSpaceValues(test_images, saveName='vae_ns_test_images_8dim')
#vae_test_9.latentSpaceValues(test_images, saveName='vae_ns_test_images_9dim')
#vae_test_10.latentSpaceValues(test_images, saveName='vae_ns_test_images_10dim')
#vae_test_11.latentSpaceValues(test_images, saveName='vae_ns_test_images_11dim')
#vae_test_12.latentSpaceValues(test_images, saveName='vae_ns_test_images_12dim')
#vae_test_13.latentSpaceValues(test_images, saveName='vae_ns_test_images_13dim')
#vae_test_14.latentSpaceValues(test_images, saveName='vae_ns_test_images_14dim')
#vae_test_15.latentSpaceValues(test_images, saveName='vae_ns_test_images_15dim')
#vae_test_16.latentSpaceValues(test_images, saveName='vae_ns_test_images_16dim')
#vae_test_18.latentSpaceValues(test_images, saveName='vae_ns_test_images_18dim')
#vae_test_20.latentSpaceValues(test_images, saveName='vae_ns_test_images_20dim')
#vae_test_25.latentSpaceValues(test_images, saveName='vae_ns_test_images_25dim')
#vae_test_30.latentSpaceValues(test_images, saveName='vae_ns_test_images_30dim')
#vae_test_40.latentSpaceValues(test_images, saveName='vae_ns_test_images_40dim')
#vae_test_50.latentSpaceValues(test_images, saveName='vae_ns_test_images_50dim')
#vae_test_75.latentSpaceValues(test_images, saveName='vae_ns_test_images_75dim')
#vae_test_100.latentSpaceValues(test_images, saveName='vae_ns_test_images_100dim')






#vae_test_6.latentSpaceValues(test_images, saveName='vae_test_images_6dim')
#vae_test_10.latentSpaceValues(test_images, saveName='vae_test_images_10dim')
#vae_test_12.latentSpaceValues(test_images, saveName='vae_test_images_12dim')
#vae_test_14.latentSpaceValues(test_images, saveName='vae_test_images_14dim',restore_encodings='vae_test_images_14dimCheckpointEncodings7250.npy', restore_loss='vae_test_images_14dimCheckpointLoss7250.npy', start=7250)

