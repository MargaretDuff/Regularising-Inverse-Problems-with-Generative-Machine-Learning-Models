# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 10:34:19 2020

@author: marga
"""
import numpy as np
from scipy import stats
import tensorflow as tf
import matplotlib.pyplot as plt


import generative_testing_class as test_class
import mnist_ae_test as ae
import mnist_gan_test as gan
import mnist_vae_test as vae

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
    print('Loaded iamges')
#ae_test=ae.mnistAE(8, 0.0,'./AE_checkpoints8_0.0/mnist_AE_8_0.0-29800')
#sae_test=ae.mnistAE(8, 1.0,'./SAE_checkpoints8_1.0/mnist_AE_8_1.0-29800')
#vae_test=vae.mnistVAE(8, './VAE_checkpoints8/mnist_VAE-29800')
#gan_test=gan.mnistGAN(16, './GAN_checkpoints16/mnist_GAN_16-39800')
#sae_50_test=ae.mnistAE(50,5.0,'./SAE_checkpoints50_5.0/mnist_AE_50_5.0-29800')
print('Loaded models')

#ae_test.encodeGenerateLoss(test_images, saveName='ae_test_images')
#sae_test.encodeGenerateLoss(test_images, saveName='sae_test_images')
#vae_test.encodeGenerateLoss(test_images, saveName='vae_test_images')
gan_test.encodeGenerateLoss(test_images, saveName='gan_test_images', restore='gan_test_imagesCheckpoint7750.npy', start=7750)
#sae_50_test.encodeGenerateLoss(test_images, saveName='SAE_50_test_images')
