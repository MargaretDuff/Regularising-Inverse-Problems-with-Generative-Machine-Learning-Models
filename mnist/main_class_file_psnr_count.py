# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 16:57:31 2020

@author: magd21
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 12:22:01 2020

@author: marga
"""

import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt


import mnist_ae_test as ae
import mnist_vae_test as vae
import mnist_gan_test_class as gan 


import inv_prob_class as IP

import optimisation_class_count as optim

import time 
test_images=np.load('./mnist_test_images.npy')
#%%
#tf.reset_default_graph()
sess = tf.InteractiveSession()


model=vae.mnistVAE(8,'./checkpoints/vae/checkpoints_no_sigmoid_8_kl_factor_0.05/checkpoints8-29800', sess)
inv_prob=IP.invProb( model, 'compressed_sensing', kernel_width=8, dim_comp=150)


np.random.seed(9)
for i in range(101,102):
    aim=test_images[i]
    inv_prob.observe_data(aim, noise_level=0.1, random_seed=i)
    opt=optim.optimisation(inv_prob, model)

    a=time.perf_counter()
    np.save('psnr_vae_hard_'+str(i)+'compressed150_fixed_count.npy', opt.gd_backtracking_z(alpha=0.030517578125, iteration_number=2000 ))
    print('VAE HARD TOOK THIS AMOUNT OF TIME    ', time.perf_counter()-a)
    a=time.perf_counter()

    np.save('psnr_vae_soft_'+str(i)+'compressed150_fixed_count.npy', opt.optim_x_soft_constraints(alpha=31.25, beta=0.030517578125*31.25,  iteration_number=2000))
    print('VAE SOFT TOOK THIS AMOUNT OF TIME    ', time.perf_counter()-a)
    a=time.perf_counter()

    np.save('psnr_vae_tik_'+str(i)+'compressed150_fixed_count.npy', opt.optim_x_tikhonov(beta=7.8125, iteration_number=10000))
    print('TIK TOOK THIS AMOUNT OF TIME    ', time.perf_counter()-a)
    



tf.reset_default_graph()

sess1 = tf.InteractiveSession()#

model=gan.mnistGAN(8,'./checkpoints/wgan/checkpoints_gan_8/mnist_GAN_8-19900' ,sess1)
inv_prob=IP.invProb( model, 'compressed_sensing', kernel_width=8, dim_comp=150)
np.random.seed(9)
for i in range(101,102):
    aim=test_images[i]
    inv_prob.observe_data(aim, noise_level=0.1, random_seed=i)
    opt=optim.optimisation(inv_prob, model)
    a=time.perf_counter()

    np.save('psnr_gan_hard_'+str(i)+'compressed150_fixed_count.npy', opt.gd_backtracking_z(alpha=500, iteration_number=2000))
    print('GAN HARD TOOK THIS AMOUNT OF TIME    ', time.perf_counter()-a)
    a=time.perf_counter()

    np.save('psnr_gan_soft_'+str(i)+'compressed150_fixed_count.npy', opt.optim_x_soft_constraints(alpha=7.8125, beta=0.15625*7.8125, iteration_number=2000))
   
    print('GAN SOFT TOOK THIS AMOUNT OF TIME    ', time.perf_counter()-a)



tf.reset_default_graph()

sess2 = tf.InteractiveSession()

model=ae.mnist_AE(8,0,'./checkpoints/ae/checkpoints8/mnist_AE_8-29800' ,sess2)
inv_prob=IP.invProb( model, 'compressed_sensing', kernel_width=8, dim_comp=150)
np.random.seed(9)
for i in range(101,102):
    aim=test_images[i]
    inv_prob.observe_data(aim, noise_level=0.1, random_seed=i)
    opt=optim.optimisation(inv_prob, model)
    a=time.perf_counter()

    np.save('psnr_ae_hard_'+str(i)+'compressed150_fixed_count.npy', opt.gd_backtracking_z(alpha=0.030517578125, iteration_number=2000))
    
    print('AE_HARD TOOK THIS AMOUNT OF TIME    ', time.perf_counter()-a)
    a=time.perf_counter()

    np.save('psnr_ae_soft_'+str(i)+'compressed150_fixed_count.npy', opt.optim_x_soft_constraints(alpha=31.25, beta=0.039062*31.25, iteration_number=2000))
   
    print('AE SOFT TOOK THIS AMOUNT OF TIME    ', time.perf_counter()-a)




