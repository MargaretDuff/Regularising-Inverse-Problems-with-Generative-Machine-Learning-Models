# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 12:22:01 2020

@author: marga
"""

import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
#import odl 


import ae_2d_shapes_class_odl as ae
import vae_2d_shapes_class_odl as vae
import gan_2d_shapes_class_odl as gan 


#train_images=np.load('../mnist_train_images.npy')
test_images=np.load('./2d_shapes_test_images2_none.npy')
test_images_bright=np.load('./2d_shapes_test_images2_circle.npy')
# for i in range(150,160):
#     plt.figure()
#     plt.imshow(test_images[i])

#%%
tf.reset_default_graph()
sess = tf.InteractiveSession()
model=vae.shapes_VAE(10, './checkpoints/vae/checkpoints_none_10_kl_factor_0.01/2d_shapes_VAE-29800',sess)



#%% Inverse problem 
import inv_prob_class as IP
#aim=np.load('aim.npy')

inv_prob=IP.invProb( model, 'tomography', dim_comp=800)

aim=test_images_bright[128]


inv_prob.observe_data(aim, noise_level=0.01)

np.save('aim_test_image_128_bright.npy',inv_prob.aim)
#inv_prob.data.show()
np.save('data_tomo_test_image_128_noise_0.01_bright.npy',inv_prob.data)


#%%
import optimisation_class_no_plots as optim

opt=optim.optimisation(inv_prob, model)
        
#opt.gd_z(0.1, np.random.normal(0,1,(1,8)))
#opt.gd_backtracking_z(0.1, np.random.normal(0,1,(1,model.n_latent)))

#opt.optim_z_sparse_deviations(0.2,0.9, np.random.normal(0,1,(1,model.n_latent)), np.random.normal(0, 0.1,(56,56)), 100)

#opt.optim_x_soft_constraints(0.5, 0.2, np.random.normal(0,1,(1,model.n_latent)), np.random.normal(0,1,(56,56) ), 100)

#plot=opt.optim_x_tik_regularisation_parameter( iteration_number=450, save_name='x_tik_128_bright', save=True, early_stop=False)
#plot=opt.optim_x_tv_regularisation_parameter( iteration_number=100, save_name='x_tv_789_bright', save=True, early_stop=False)

#plot=opt.gd_z_regularisation_parameter(initial_z=np.random.normal(0,1,(4,model.n_latent)), iteration_number=40, save_name='VAE_10dim_kl_factor_0.01_no_sigmoid_z_optimisation_128_bright', save =True)

#plot=opt.optim_z_sparse_regularisation_parameter(initial_z=np.random.normal(0,1,(4,model.n_latent)), initial_u=np.zeros((4,56,56)), iteration_number=35,  save_name='VAE_10dim_kl_factor_no_sigmoid_z_sparse_128_bright', early_stop=False, save=True)
plot=opt.optim_x_soft_constraints_regularisation_parameter(initial_z=np.random.normal(0,1,(4,model.n_latent)), initial_x=np.random.normal(0,1,(4,56,56)), iteration_number=800, save_name='VAE_10dim_kl_factor_no_sigmoid_x_soft_128_bright', early_stop=False, save=True)


#%%


#%%
tf.reset_default_graph()
sess = tf.InteractiveSession()


model=ae.shapes_AE(10, '../../2d_shapes_ae/checkpoints_no_sigmoid/checkpoints_none_10/2d_shapes_AE-29800', sess)

#import optimisation_class_no_plots as optim    



#opt=optim.optimisation(inv_prob, model)


#plot, =opt.optim_x_tik_regularisation_parameter( iteration_number=250, save_name='AE_10dim_no_sigmoid_x_tik_one', save=True, early_stop=False)
#plot=opt.optim_x_tv_regularisation_parameter( iteration_number=100, save_name='x_tv_789_bright', save=True, early_stop=False)


#plot=opt.gd_z_regularisation_parameter(initial_z=np.random.normal(0,1,(4,model.n_latent)), iteration_number=40, save_name='AE_10dim_no_sigmoid_z_optimisation', save =True)
#plot=opt.optim_x_soft_constraints_regularisation_parameter(initial_z=np.random.normal(0,1,(4,model.n_latent)), initial_x=np.random.normal(0,1,(4,56,56)), iteration_number=40, save_name='AE_10dim_no_sigmoid_x_soft', early_stop=False, save=True)
#plot=opt.optim_z_sparse_regularisation_parameter(initial_z=np.random.normal(0,1,(4,model.n_latent)), initial_u=np.zeros((4,56,56)), iteration_number=40,  save_name='AE_10dim_no_sigmoid_z_sparse', early_stop=False, save=True)






#%%
#tf.reset_default_graph()
#sess = tf.InteractiveSession()
#model=vae.shapes_VAE(10, '../checkpoints_no_sigmoid/checkpoints_none_10/2d_shapes_VAE-29800',sess)
#model=gan.shapesGAN(10, '../../2d_shapes_wgan/checkpoints_ns_10_none/2d_shapes_wgan_10dim-29800',sess)
   



#opt=optim.optimisation(inv_prob, model)
        

#plot, fig, fig2=opt.optim_x_tik_regularisation_parameter( iteration_number=250, save_name='GAN_10dim_no_sigmoid_x_tik_one', save=True, early_stop=False)
#plot=opt.gd_z_regularisation_parameter(initial_z=np.random.normal(0,1,(4,model.n_latent)), iteration_number=40, save_name='GAN_10dim_no_sigmoid_z_optimisation', save =True)

#plot=opt.optim_z_sparse_regularisation_parameter(initial_z=np.random.normal(0,1,(4,model.n_latent)), initial_u=np.zeros((4,56,56)), iteration_number=40,  save_name='GAN_10dim_no_sigmoid_z_sparse', early_stop=False, save=True)

#plot=opt.optim_x_soft_constraints_regularisation_parameter(initial_z=np.random.normal(0,1,(4,model.n_latent)), initial_x=np.random.normal(0,1,(4,56,56)), iteration_number=40, save_name='GAN_10dim_no_sigmoid_x_soft', early_stop=False, save=True)

#%%
import inv_prob_class as IP
#aim=np.load('aim.npy')

inv_prob=IP.invProb( model, 'compressed_sensing', dim_comp=300)


#aim=test_images[153] #The number 5 with a tick

aim=test_images[2379] 
np.random.seed(9)
#aim=model.generator(np.random.normal(0,1, (1,model.n_latent) ))

inv_prob.observe_data(aim, noise_level=0.1)
np.save('aim_shapes_2379.npy', inv_prob.aim)
np.save('data_shapes_cs_2379_0.1_noise_300_dim.npy', inv_prob.data)
#%%
import optimisation_class_no_plots as optim

opt=optim.optimisation(inv_prob, model)
#for i in range(10):
#    plot=opt.gd_z_regularisation_parameter(initial_z=np.random.normal(0,1,(1,model.n_latent)), iteration_number=30, save_name='AE_10_dim_shapes_cs_2379_z_opt_iter_'+str(i), save =True)
#    plot=opt.optim_z_sparse_regularisation_parameter(initial_z=np.random.normal(0,1,(1,model.n_latent)), initial_u=np.zeros((1,56,56)), iteration_number=30, mu_min=0.45, mu_max=0.5,  save_name='AE_10_dim_shapes_cs_2379_z_sparse_iter_'+str(i), early_stop=False, save=True)
#    plot=opt.optim_x_soft_constraints_regularisation_parameter(initial_z=np.random.normal(0,1,(1,model.n_latent)), initial_x=np.random.normal(0,1,(1,56,56)), iteration_number=30, mu_min=0.45, mu_max=0.5, save_name='AE_10_dim_shapes_cs_2379_x_soft_iter_'+str(i),early_stop=False, save=True)
    #plot=opt.optim_x_tik_regularisation_parameter( iteration_number=1500, save_name='x_tik_shapes_cs_789', save=True, early_stop=False)
#    plot=opt.optim_x_tv_regularisation_parameter( iteration_number=100, save_name='x_tv_shapes_cs_2379', save=True, early_stop=False)

#
