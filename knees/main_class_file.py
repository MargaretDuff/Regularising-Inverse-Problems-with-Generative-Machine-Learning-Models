# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 12:22:01 2020

@author: marga
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



import knees_vae_test as vae
#import mnist_gan_test as gan 

test_images=np.load('./knee_fastMRI_test.npy')
for i in range(150,160):
    plt.figure()
    plt.imshow(test_images[i])


#%%
tf.reset_default_graph()
sess = tf.InteractiveSession()
model=vae.kneesVAE(180,'./checkpoints2_180_kl_factor_0.0125/knees_VAE-399800',sess )
# test.elephant()


for i in range(16):
    a=model.generator(np.random.normal(0,1, (1,model.n_latent) ))
    a.show()
    #cv2.imwrite('check_generate_none'+str(i)+'.png',a[i]*256)
    
#model.generator(model.encoder(model.x_space.element(test_images[153]))).show()
#%% Inverse problem 
import inv_prob_class as IP
#aim=np.load('aim.npy')

inv_prob=IP.invProb( model, 'convolution')



aim=test_images[135] 
np.random.seed(9)
#aim=model.generator(np.random.normal(0,1, (1,model.n_latent) ))


inv_prob.observe_data(aim, noise_level=0.05)
inv_prob.aim.show()
inv_prob.data.show()

#for inpaiting only
#inv_prob.inv_flatten(inv_prob.data).show()

# works only if x_space=y_space
#print(inv_prob.data_discrepancy(inv_prob.data)==0) 
print(np.sqrt(inv_prob.data_discrepancy(inv_prob.aim)))

#%%
import optimisation_class as optim

opt=optim.optimisation(inv_prob, model)
        
# opt.optim_x_tikhonov(0.2)
# #opt.optim_x_tv(0.5)

#z=opt.gd_backtracking_z(0.1, np.random.normal(0,1,(8,model.n_latent)))

opt.optim_z_sparse_deviations(0.05,0.3, np.random.normal(0,1,(8,model.n_latent)), 0.01*np.random.normal(0,1,(8,80,80)), 120)

x,z=opt.optim_x_soft_constraints(0.2, 2, np.random.normal(0,1,(8,model.n_latent)), np.random.normal(0,1,(8,80,80) ), 70)

#fig,fig2,plot=opt.gd_z_regularisation_parameter(initial_z=np.random.normal(0,1,(4,model.n_latent)), iteration_number=20, save_name='AE_8_dim_no_sigmoid_z_optimisation_one_', save =True)
#plot, fig, fig2=opt.optim_z_sparse_regularisation_parameter(initial_z=np.random.normal(0,1,(4,model.n_latent)), initial_u=np.zeros((4,28,28)), iteration_number=20,  save_name='AE_8_dim_no_sigmoid_z_sparse_one_', early_stop=False, save=True)
#plot, fig, fig2=opt.optim_x_soft_constraints_regularisation_parameter(initial_z=np.random.normal(0,1,(4,model.n_latent)), initial_x=np.random.normal(0,1,(4,28,28)), iteration_number=20, save_name='AE_8_dim_no_sigmoid_x_soft_one', early_stop=False, save=True)
#plot, fig, fig2=opt.optim_x_tik_regularisation_parameter( iteration_number=1500, save_name='AE_8_dim_no_sigmoid_x_tik_one', save=True, early_stop=False)
#       
#%% Where there is an encoder
# import optimisation_class as optim

# opt=optim.optimisation(inv_prob, model)
        
# #opt.gd_z(0.1, np.random.normal(0,1,(1,model.n_latent)))
opt.gd_backtracking_z(1, model.encoder(inv_prob.data))

z,u=opt.optim_z_sparse_deviations(1,0.1, model.encoder(inv_prob.data), 0.01*np.random.normal(0,1,(80,80)), 100)

opt.optim_x_soft_constraints(1, 2, model.encoder(inv_prob.data), model.generator(model.encoder(inv_prob.data)) ,200)


#%%
