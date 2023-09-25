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


import ae_2d_shapes_class_odl as ae
import vae_2d_shapes_class_odl as vae
import gan_2d_shapes_class_odl as gan 
import time 


import inv_prob_class as IP

import optimisation_class_count as optim

test_images=np.load('./2d_shapes_test_images2_none.npy')

#%%
tf.reset_default_graph()
sess = tf.InteractiveSession()


model=vae.shapes_VAE(10, './checkpoints/vae/checkpoints_none_10_kl_factor_0.01/2d_shapes_VAE-29800',sess)
inv_prob=IP.invProb( model, 'tomography')


np.random.seed(9)
for i in range(102,103):
    aim=test_images[i]
    inv_prob.observe_data(aim, noise_level=0.1, random_seed=i)
    opt=optim.optimisation(inv_prob, model)
    a=time.perf_counter()

    np.save('psnr_vae_hard_'+str(i)+'more_noise_fixed_count.npy', opt.gd_backtracking_z(alpha=0.0244140625, iteration_number=2000 ))
    print('VAE HARD TOOK THIS AMOUNT OF TIME    ', time.perf_counter()-a)
    a=time.perf_counter()

    np.save('psnr_vae_soft_'+str(i)+'more_noise_fixed_count.npy', opt.optim_x_soft_constraints(alpha=0.390625, beta=0.390625*0.625,  iteration_number=2000))

    print('VAE SOFT TOOK THIS AMOUNT OF TIME    ', time.perf_counter()-a)
    a=time.perf_counter()

    np.save('psnr_vae_tik_'+str(i)+'more_noise_fixed_count.npy', opt.optim_x_tikhonov(beta=0.1953125, iteration_number=10000))
    print('TIK TOOK THIS AMOUNT OF TIME    ', time.perf_counter()-a)




tf.reset_default_graph()

sess1 = tf.InteractiveSession()#

model=gan.shapesGAN(10, './checkpoints/wgan/checkpoints_ns_10_none/2d_shapes_wgan_10dim-29800',sess1)

inv_prob=IP.invProb( model, 'tomography')
np.random.seed(9)
for i in range(102,103):
    aim=test_images[i]
    inv_prob.observe_data(aim, noise_level=0.1, random_seed=i)
    opt=optim.optimisation(inv_prob, model)
    a=time.perf_counter()
    
    np.save('psnr_gan_hard_'+str(i)+'more_noise_fixed_count.npy', opt.gd_backtracking_z(alpha=0.78125, iteration_number=2000))
    
    print('GAN HARD TOOK THIS AMOUNT OF TIME    ', time.perf_counter()-a)
    a=time.perf_counter()

    np.save('psnr_gan_soft_'+str(i)+'more_noise_fixed_count.npy', opt.optim_x_soft_constraints(alpha=0.09765625, beta=0.09765625*10, iteration_number=2000))
    print('GAN SOFT TOOK THIS AMOUNT OF TIME    ', time.perf_counter()-a)





tf.reset_default_graph()

sess2 = tf.InteractiveSession()

model=ae.shapes_AE(10, './checkpoints/ae/checkpoints_none_10/2d_shapes_AE-29800', sess2)
inv_prob=IP.invProb( model, 'tomography')
np.random.seed(9)
for i in range(102,103):
    aim=test_images[i]
    inv_prob.observe_data(aim, noise_level=0.1, random_seed=i)
    opt=optim.optimisation(inv_prob, model)
    a=time.perf_counter()

    np.save('psnr_ae_hard_'+str(i)+'more_noise_fixed_count.npy', opt.gd_backtracking_z(alpha=0.00152587890625, iteration_number=2000))
    print('AE_HARD TOOK THIS AMOUNT OF TIME    ', time.perf_counter()-a)
    a=time.perf_counter()

    np.save('psnr_ae_soft_'+str(i)+'more_noise_fixed_count.npy', opt.optim_x_soft_constraints(alpha=0.390625, beta=0.39062*0.15625, iteration_number=2000))
    print('AE SOFT TOOK THIS AMOUNT OF TIME    ', time.perf_counter()-a)





