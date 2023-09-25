# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 10:34:19 2020

@author: marga
"""
import numpy as np
#from scipy import stats
import tensorflow as tf
#import matplotlib.pyplot as plt


import generative_testing_class as test_class
import mnist_ae_test as ae
import mnist_gan_test as gan
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
    train_images=np.load('mnist_train_images.npy')
    test_images=np.load('mnist_test_images.npy')
    print('Loaded images')


#for i in [9,10,11,12,13,14,15,16,18,20,25]:

 #   gan_test=gan.mnistGAN(i ,'../mnist_gan/checkpoints_ns/checkpoints_gan_'+str(i)+'/mnist_GAN_'+str(i)+'-19900')
  #  gan_test.projectManifold(test_images, saveName='gan_ns_test_images_'+str(i)+'dim')


for i in [5,6,7,8,9,10,11,12,13,14,15,16,18,20,25]:
#for i in [7]:

    ae_test=ae.mnistAE(i ,'../ae_mnist/checkpoints_ns/checkpoints'+str(i)+'/mnist_AE_'+str(i)+'-29800')
    ae_test.projectManifold(test_images,noise_level=0.05, saveName='ae_ns_test_images_'+str(i)+'dim_gauss0.05_')


for i in [5,6,7,8,9,10,11,12,13,14,15,16,18,20,25]:
#for i in [7]:

    vae_test=vae.mnistVAE(i ,'../mnist_vae/checkpoints_no_sigmoid_'+str(i)+'/checkpoints'+str(i)+'-29800')
    vae_test.projectManifold(test_images,noise_level=0.05, saveName='vae_ns_test_images_'+str(i)+'dim_gauss0.05_')

#for i in [5,6,7,8,9,10,11,12,13,14,15,16,18,20,25]:

 #   gan_test=gan.mnistGAN(i ,'../mnist_gan/checkpoints_ns/checkpoints_gan_'+str(i)+'/mnist_GAN_'+str(i)+'-19900')
  #  gan_test.projectManifold(test_images, noise_level=0.1, saveName='gan_ns_test_images_'+str(i)+'dim_gauss0.1_')

