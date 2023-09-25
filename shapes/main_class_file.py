# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 12:22:01 2020

@author: marga
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import odl 
import cv2


import wgan_2d_shapes_class as wgan
import vae_2d_shapes_class as vae 
import inv_prob_class as IP
import optimisation_class as optim

#%%
tf.reset_default_graph()
sess = tf.InteractiveSession()
test=wgan.shapesGAN(12, './wgan_checkpoints_12/checkpoints_12_none/2d_shapes_wgan_12dim-29800',sess)
#test=vae.shapes_VAE(10,'./vae_checkpoints_10/checkpoints_none_12/2d_shapes_VAE-29800',sess)
# test.elephant()
a=test.generate(np.random.normal(0,1, (16,12) ))
import matplotlib.pyplot as plt
for i in range(16):
    plt.figure()
    plt.imshow(a[i]*256, cmap='gray')
    #cv2.imwrite('check_generate_none'+str(i)+'.png',a[i]*256)
    
aim=np.load('aim.npy')
plt.figure()
plt.imshow(aim)
inv_prob=IP.invProb(aim, 'denoising')




#%%
np.random.seed(42)
opt=optim.optimisation(inv_prob, test,sess)
#opt.optim_z(0.1, np.random.normal(0,1,(16,10)), 100, plot=True)

#opt.optim_z_sparse_deviations(0.1,0.3, np.random.normal(0,1,(16,12)), np.zeros((16,56,56)), 100, plot=True)

#opt.optim_z_odlgd(0.1,np.random.normal(0,1,(1,12)))

opt.palm_z_u(0.1,0.3, np.random.normal(0,1,(16,12)), np.zeros((16,56,56)), iteration_number=100, plot=True, gamma=2)