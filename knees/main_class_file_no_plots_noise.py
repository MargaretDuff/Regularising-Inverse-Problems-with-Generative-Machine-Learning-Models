# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt


import knees2_128_128_vae_test as vae


test_images=np.load('./datasets/knee_fastMRI_test_128_cleaned.npy')


#%%
tf.reset_default_graph()
sess = tf.InteractiveSession()
model=vae.kneesVAE(800,'./checkpoints/checkpoints_no_sigmoid/checkpoints2_128_cleaned_800_kl_factor_0.001/knees_VAE-249800',sess )


for image in [270,380,580]:
    #%% Inverse problem
    import inv_prob_class as IP
    for noise_factor in range(5):


        inv_prob=IP.invProb( model,  'tomography')

        image_no=image
        noise_level=0.2*(1/(2**noise_factor))

        aim=test_images[image_no]
        np.random.seed(9)
        #aim=model.generator(np.random.normal(0,1, (1,model.n_latent) ))




        inv_prob.observe_data(aim, noise_level=noise_level, random_seed=image_no)
     

        np.save('./inverse_problems/noise_level/aim_knees_test_esc_'+str(image_no)+'.npy', inv_prob.aim)
        np.save('./inverse_problems/noise_level/data_knees_mri_'+str(image_no)+'_noise_'+str(noise_level)+'.npy', inv_prob.data)
        np.save('./inverse_problems/noise_level/adjoint_reconstruction_knees_mri_'+str(image_no)+'_noise_'+str(noise_level)+'.npy', inv_prob.A.adjoint(inv_prob.data))
        adjoint=inv_prob.A.adjoint(inv_prob.data)
        #%%
        import optimisation_class_no_plots as optim

        opt=optim.optimisation(inv_prob, model)


        plot=opt.gd_z_regularisation_parameter(initial_z=np.random.normal(0,1,(1,opt.n_latent)), iteration_number=100, save_name='./inverse_problems/noise_level/VAE2_128_128_800_0.001_z_optimisation_knees_test_cleaned_no_sig'+str(image_no)+'_noise_'+str(noise_level), save =True, alpha_factor=0.05)
        plot=opt.optim_z_sparse_regularisation_parameter(initial_z=np.random.normal(0,1,(1,opt.n_latent)), initial_u=np.zeros((4,128,128)), iteration_number=100,  save_name='./inverse_problems/noise_level/VAE2_128_128_800_0.001_z_sparse_knees_test_cleaned_no_sig'+str(image_no)+'_noise_'+str(noise_level), early_stop=False, save=True, lambda_factor=0.05, mu_min=0.4, mu_max=0.5)
        plot=opt.optim_x_soft_constraints_regularisation_parameter(initial_z=np.random.normal(0,1,(1,opt.n_latent)), initial_x=np.random.normal(0,1,(1,128,128)), iteration_number=100, save_name='./inverse_problems/noise_level/VAE2_128_128_800_0.001_x_soft_knees_test_cleaned_no_sig'+str(image_no)+'_noise_'+str(noise_level), early_stop=False, save=True,  lambda_factor=0.05, mu_min=0.4, mu_max=0.5)
        plot=opt.optim_x_tik_regularisation_parameter( iteration_number=100, save_name='./inverse_problems/noise_level/x_tik_knees_test_cleaned_no_sig'+str(image_no)+'_noise_'+str(noise_level), save=True, early_stop=False, beta_factor=0.05)

        
