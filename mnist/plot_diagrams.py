# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 10:56:27 2020

@author: marga
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# SMALL_SIZE = 8
# MEDIUM_SIZE = 10
# BIGGER_SIZE = 12

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# # #%%

# # ae_x_tik=np.load('AE_8_dim_no_sigmoid_x_tik_oneloss.npy')
# ae_x_soft=np.load('AE_8_dim_no_sigmoid_x_soft_onelambda_mu_loss.npy')
# ae_z_sparse=np.load('AE_8_dim_no_sigmoid_z_sparse_one_lambda_mu_loss.npy')
# ae_z_opt=np.load('AE_8_dim_no_sigmoid_z_optimisation_one__loss.npy')



# # Plot figure with subplots of different sizes
# fig = plt.figure(1)
# # set up subplot grid
# gridspec.GridSpec(3,4)

# # large subplot
# plt.subplot2grid((3,4), (0,0), colspan=2, rowspan=2)
# plot=ae_z_opt
# plt.scatter(plot[:,0], plot[:,1], s=10)
# plt.hlines(28*28*0.1**2, np.min(plot[:,0]), np.max(plot[:,0]))
# plt.xscale('log')
# plt.xlabel('lambda')
# plt.ylabel('Data discrepancy')
# plt.ylim(0,30)
# #.colorbar()

# # small subplot 1
# plt.subplot2grid((3,4), (0,2))
# plt.imshow(np.load('AE_8_dim_no_sigmoid_z_optimisation_one_lambda_2.5.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# plt.yticks([])
# plt.xticks([])
# plt.xlabel('lambda={:.2E}'.format(2.5))

# # small subplot 2
# plt.subplot2grid((3,4), (0,3))
# plt.imshow(np.load('AE_8_dim_no_sigmoid_z_optimisation_one_lambda_0.15625.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# plt.yticks([])
# plt.xticks([])
# plt.xlabel('lambda={:.2E}'.format(0.15625))


# # small subplot 3
# plt.subplot2grid((3,4), (1,2))
# plt.imshow(np.load('AE_8_dim_no_sigmoid_z_optimisation_one_lambda_0.01953125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# plt.yticks([])
# plt.xticks([])
# plt.xlabel('lambda={:.2E}'.format(0.01953125))


# # small subplot 4
# plt.subplot2grid((3,4), (1,3))
# plt.imshow(np.load('AE_8_dim_no_sigmoid_z_optimisation_one_lambda_0.0048828125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# plt.yticks([])
# plt.xticks([])
# plt.xlabel('lambda={:.2E}'.format(0.0048828125))



# # fit subplots and save fig
# fig.tight_layout()
# fig.set_size_inches(w=11,h=6)
# fig_name = 'ae_z_opt_one.png'
# fig.savefig(fig_name, bbox_inches='tight')

# #%%

# # Plot figure with subplots of different sizes
# fig = plt.figure(1)
# # set up subplot grid
# gridspec.GridSpec(3,4)

# # large subplot
# plt.subplot2grid((3,4), (0,0), colspan=2, rowspan=2)
# plot=ae_x_tik
# plt.scatter(plot[:,0], plot[:,1], s=10)
# plt.hlines(28*28*0.1**2, np.min(plot[:,0]), np.max(plot[:,0]))
# plt.xscale('log')
# plt.xlabel('lambda')
# plt.ylabel('Data discrepancy')
# plt.ylim(0,30)
# #plt.colorbar()

# # small subplot 1
# plt.subplot2grid((3,4), (0,2))
# plt.imshow(np.load('AE_8_dim_no_sigmoid_x_tik_one_lambda_2.5.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# plt.yticks([])
# plt.xticks([])
# plt.xlabel('lambda={:.2E}'.format(2.5))

# # small subplot 2
# plt.subplot2grid((3,4), (0,3))
# plt.imshow(np.load('AE_8_dim_no_sigmoid_x_tik_one_lambda_0.15625.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# plt.yticks([])
# plt.xticks([])
# plt.xlabel('lambda={:.2E}'.format(0.15625))


# # small subplot 3
# plt.subplot2grid((3,4), (1,2))
# plt.imshow(np.load('AE_8_dim_no_sigmoid_x_tik_one_lambda_0.01953125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# plt.yticks([])
# plt.xticks([])
# plt.xlabel('lambda={:.2E}'.format(0.01953125))


# # small subplot 4
# plt.subplot2grid((3,4), (1,3))
# plt.imshow(np.load('AE_8_dim_no_sigmoid_x_tik_one_lambda_0.0048828125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# plt.yticks([])
# plt.xticks([])
# plt.xlabel('lambda={:.2E}'.format(0.0048828125))



# # fit subplots and save fig
# fig.tight_layout()
# fig.set_size_inches(w=11,h=6)
# fig_name = 'ae_x_tik_one.png'
# fig.savefig(fig_name, bbox_inches='tight')

# #%%

# # Plot figure with subplots of different sizes
# fig = plt.figure(1)
# # set up subplot grid
# gridspec.GridSpec(3,4)

# # large subplot
# plt.subplot2grid((3,4), (0,0), colspan=2, rowspan=2)
# plot=ae_x_soft
# plt.scatter(plot[:,0], plot[:,2], c=plot[:,1],s=10)
# plt.colorbar()
# plt.hlines(28*28*0.1**2, np.min(plot[:,0]), np.max(plot[:,0]))
# plt.xscale('log')
# plt.xlabel('lambda')
# plt.ylabel('Data discrepancy')
# plt.ylim(0,30)


# # small subplot 1
# plt.subplot2grid((3,4), (0,2))
# plt.imshow(np.load('AE_8_dim_no_sigmoid_x_soft_onelambda_2.5_mu_2.5.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# plt.yticks([])
# plt.xticks([])
# plt.xlabel('lambda={:.2E},\n $\mu$={:.2E}'.format(2.5,2.5))

# # small subplot 2
# plt.subplot2grid((3,4), (0,3))
# plt.imshow(np.load('AE_8_dim_no_sigmoid_x_soft_onelambda_0.15625_mu_0.3125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# plt.yticks([])
# plt.xticks([])
# plt.xlabel('lambda={:.2E},\n $\mu$={:.2E}'.format(0.15625,0.3125))


# # small subplot 3
# plt.subplot2grid((3,4), (1,2))
# plt.imshow(np.load('AE_8_dim_no_sigmoid_x_soft_onelambda_0.01953125_mu_0.078125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# plt.yticks([])
# plt.xticks([])
# plt.xlabel('lambda={:.2E},\n  $\mu$={:.2E}'.format(0.01953125,0.078125))


# # small subplot 4
# plt.subplot2grid((3,4), (1,3))
# plt.imshow(np.load('AE_8_dim_no_sigmoid_x_soft_onelambda_0.0048828125_mu_0.01953125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# plt.yticks([])
# plt.xticks([])
# plt.xlabel('lambda={:.2E},\n  $\mu$={:.2E}'.format(0.0048828125,0.01953125))



# # fit subplots and save fig
# fig.tight_layout()
# fig.set_size_inches(w=11,h=6)
# fig_name = 'ae_x_soft_one.png'
# fig.savefig(fig_name, bbox_inches='tight')

# #%%

# # Plot figure with subplots of different sizes
# fig = plt.figure(1)
# # set up subplot grid
# gridspec.GridSpec(3,4)

# # large subplot
# plt.subplot2grid((3,4), (0,0), colspan=2, rowspan=2)
# plot=ae_z_sparse
# plt.scatter(plot[:,0], plot[:,2], c=plot[:,1],s=10)
# plt.colorbar()
# plt.hlines(28*28*0.1**2, np.min(plot[:,0]), np.max(plot[:,0]))
# plt.xscale('log')
# plt.xlabel('lambda')
# plt.ylabel('Data discrepancy')
# plt.ylim(0,30)



# # small subplot 1
# plt.subplot2grid((3,4), (0,2))
# plt.imshow(np.load('AE_8_dim_no_sigmoid_z_sparse_one_lambda_2.5_mu_2.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# plt.yticks([])
# plt.xticks([])
# plt.xlabel('lambda={:.2E},\n  $\mu$={:.2E}'.format(2.5,2.5))

# # small subplot 2
# plt.subplot2grid((3,4), (0,3))
# plt.imshow(np.load('AE_8_dim_no_sigmoid_z_sparse_one_lambda_0.15625_mu_0.25.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# plt.yticks([])
# plt.xticks([])
# plt.xlabel('lambda={:.2E},\n  $\mu$={:.2E}'.format(0.15625,0.25))


# # small subplot 3
# plt.subplot2grid((3,4), (1,2))
# plt.imshow(np.load('AE_8_dim_no_sigmoid_z_sparse_one_lambda_0.01953125_mu_0.03125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# plt.yticks([])
# plt.xticks([])
# plt.xlabel('lambda={:.2E},\n $\mu$={:.2E}'.format(0.01953125,0.03125))


# # small subplot 4
# plt.subplot2grid((3,4), (1,3))
# plt.imshow(np.load('AE_8_dim_no_sigmoid_z_sparse_one_lambda_0.0048828125_mu_0.015625.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# plt.yticks([])
# plt.xticks([])
# plt.xlabel('lambda={:.2E}, \n $\mu$={:.2E}'.format(0.0048828125,0.015625))



# # fit subplots and save fig
# fig.tight_layout()
# fig.set_size_inches(w=11,h=6)
# fig_name = 'ae_z_sparse_one.png'
# fig.savefig(fig_name, bbox_inches='tight')







#%%
#%%


# vae_x_soft=np.load('VAE_8_dim_no_sigmoid_x_soft_onelambda_mu_loss.npy')
# vae_z_sparse=np.load('VAE_8_dim_no_sigmoid_z_sparse_one_lambda_mu_loss.npy')
# vae_z_opt=np.load('VAE_8_dim_no_sigmoid_z_optimisation_one__loss.npy')



# # Plot figure with subplots of different sizes
# fig = plt.figure(1)
# # set up subplot grid
# gridspec.GridSpec(3,4)

# # large subplot
# plt.subplot2grid((3,4), (0,0), colspan=2, rowspan=2)
# plot=vae_z_opt
# plt.scatter(plot[:,0], plot[:,1], s=10)
# plt.hlines(28*28*0.1**2, np.min(plot[:,0]), np.max(plot[:,0]))
# plt.xscale('log')
# plt.xlabel('lambda')
# plt.ylabel('Data discrepancy')
# plt.ylim(0,30)
# #.colorbar()

# # small subplot 1
# plt.subplot2grid((3,4), (0,2))
# plt.imshow(np.load('VAE_8_dim_no_sigmoid_z_optimisation_one_lambda_2.5.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# plt.yticks([])
# plt.xticks([])
# plt.xlabel('lambda={:.2E}'.format(2.5))

# # small subplot 2
# plt.subplot2grid((3,4), (0,3))
# plt.imshow(np.load('VAE_8_dim_no_sigmoid_z_optimisation_one_lambda_0.15625.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# plt.yticks([])
# plt.xticks([])
# plt.xlabel('lambda={:.2E}'.format(0.15625))


# # small subplot 3
# plt.subplot2grid((3,4), (1,2))
# plt.imshow(np.load('VAE_8_dim_no_sigmoid_z_optimisation_one_lambda_0.01953125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# plt.yticks([])
# plt.xticks([])
# plt.xlabel('lambda={:.2E}'.format(0.01953125))


# # small subplot 4
# plt.subplot2grid((3,4), (1,3))
# plt.imshow(np.load('VAE_8_dim_no_sigmoid_z_optimisation_one_lambda_0.0048828125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# plt.yticks([])
# plt.xticks([])
# plt.xlabel('lambda={:.2E}'.format(0.0048828125))



# # fit subplots and save fig
# fig.tight_layout()
# fig.set_size_inches(w=11,h=6)
# fig_name = 'vae_z_opt_one.png'
# fig.savefig(fig_name, bbox_inches='tight')

# #%%
# # Plot figure with subplots of different sizes
# fig = plt.figure(1)
# # set up subplot grid
# gridspec.GridSpec(3,4)

# # large subplot
# plt.subplot2grid((3,4), (0,0), colspan=2, rowspan=2)
# plot=vae_x_soft
# plt.scatter(plot[:,0], plot[:,2], c=plot[:,1],s=10)
# plt.colorbar()
# plt.hlines(28*28*0.1**2, np.min(plot[:,0]), np.max(plot[:,0]))
# plt.xscale('log')
# plt.xlabel('lambda')
# plt.ylabel('Data discrepancy')
# plt.ylim(0,30)


# # small subplot 1
# plt.subplot2grid((3,4), (0,2))
# plt.imshow(np.load('VAE_8_dim_no_sigmoid_x_soft_onelambda_2.5_mu_2.5.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# plt.yticks([])
# plt.xticks([])
# plt.xlabel('lambda={:.2E},\n $\mu$={:.2E}'.format(2.5,2.5))

# # small subplot 2
# plt.subplot2grid((3,4), (0,3))
# plt.imshow(np.load('VAE_8_dim_no_sigmoid_x_soft_onelambda_0.15625_mu_0.3125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# plt.yticks([])
# plt.xticks([])
# plt.xlabel('lambda={:.2E},\n $\mu$={:.2E}'.format(0.15625,0.3125))


# # small subplot 3
# plt.subplot2grid((3,4), (1,2))
# plt.imshow(np.load('VAE_8_dim_no_sigmoid_x_soft_onelambda_0.01953125_mu_0.078125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# plt.yticks([])
# plt.xticks([])
# plt.xlabel('lambda={:.2E},\n  $\mu$={:.2E}'.format(0.01953125,0.078125))


# # small subplot 4
# plt.subplot2grid((3,4), (1,3))
# plt.imshow(np.load('VAE_8_dim_no_sigmoid_x_soft_onelambda_0.0048828125_mu_0.01953125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# plt.yticks([])
# plt.xticks([])
# plt.xlabel('lambda={:.2E},\n  $\mu$={:.2E}'.format(0.0048828125,0.01953125))



# # fit subplots and save fig
# fig.tight_layout()
# fig.set_size_inches(w=11,h=6)
# fig_name = 'vae_x_soft_one.png'
# fig.savefig(fig_name, bbox_inches='tight')

# #%%

# # Plot figure with subplots of different sizes
# fig = plt.figure(1)
# # set up subplot grid
# gridspec.GridSpec(3,4)

# # large subplot
# plt.subplot2grid((3,4), (0,0), colspan=2, rowspan=2)
# plot=vae_z_sparse
# plt.scatter(plot[:,0], plot[:,2], c=plot[:,1],s=10)
# plt.colorbar()
# plt.hlines(28*28*0.1**2, np.min(plot[:,0]), np.max(plot[:,0]))
# plt.xscale('log')
# plt.xlabel('lambda')
# plt.ylabel('Data discrepancy')
# plt.ylim(0,30)


# # small subplot 1
# plt.subplot2grid((3,4), (0,2))
# plt.imshow(np.load('VAE_8_dim_no_sigmoid_z_sparse_one_lambda_2.5_mu_2.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# plt.yticks([])
# plt.xticks([])
# plt.xlabel('lambda={:.2E},\n  $\mu$={:.2E}'.format(2.5,2.5))

# # small subplot 2
# plt.subplot2grid((3,4), (0,3))
# plt.imshow(np.load('VAE_8_dim_no_sigmoid_z_sparse_one_lambda_0.15625_mu_0.25.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# plt.yticks([])
# plt.xticks([])
# plt.xlabel('lambda={:.2E},\n  $\mu$={:.2E}'.format(0.15625,0.25))


# # small subplot 3
# plt.subplot2grid((3,4), (1,2))
# plt.imshow(np.load('VAE_8_dim_no_sigmoid_z_sparse_one_lambda_0.01953125_mu_0.03125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# plt.yticks([])
# plt.xticks([])
# plt.xlabel('lambda={:.2E},\n $\mu$={:.2E}'.format(0.01953125,0.03125))


# # small subplot 4
# plt.subplot2grid((3,4), (1,3))
# plt.imshow(np.load('VAE_8_dim_no_sigmoid_z_sparse_one_lambda_0.0048828125_mu_0.015625.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# plt.yticks([])
# plt.xticks([])
# plt.xlabel('lambda={:.2E}, \n $\mu$={:.2E}'.format(0.0048828125,0.015625))



# # fit subplots and save fig
# fig.tight_layout()
# fig.set_size_inches(w=11,h=6)
# fig_name = 'vae_z_sparse_one.png'
# fig.savefig(fig_name, bbox_inches='tight')




#%%


# gan_x_soft=np.load('GAN_8_dim_no_sigmoid_x_soft_onelambda_mu_loss.npy')
# gan_z_sparse=np.load('GAN_8_dim_no_sigmoid_z_sparse_one_lambda_mu_loss.npy')
# gan_z_opt=np.load('GAN_8_dim_no_sigmoid_z_optimisation_one__loss.npy')



# # Plot figure with subplots of different sizes
# fig = plt.figure(1)
# # set up subplot grid
# gridspec.GridSpec(3,4)

# # large subplot
# plt.subplot2grid((3,4), (0,0), colspan=2, rowspan=2)
# plot=gan_z_opt
# plt.scatter(plot[:,0], plot[:,1], s=10)
# plt.hlines(28*28*0.1**2, np.min(plot[:,0]), np.max(plot[:,0]))
# plt.xscale('log')
# plt.xlabel('lambda')
# plt.ylabel('Data discrepancy')
# plt.ylim(0,30)
# #.colorbar()

# # small subplot 1
# plt.subplot2grid((3,4), (0,2))
# plt.imshow(np.load('GAN_8_dim_no_sigmoid_z_optimisation_one_lambda_2.5.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# plt.yticks([])
# plt.xticks([])
# plt.xlabel('lambda={:.2E}'.format(2.5))

# # small subplot 2
# plt.subplot2grid((3,4), (0,3))
# plt.imshow(np.load('GAN_8_dim_no_sigmoid_z_optimisation_one_lambda_0.15625.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# plt.yticks([])
# plt.xticks([])
# plt.xlabel('lambda={:.2E}'.format(0.15625))


# # small subplot 3
# plt.subplot2grid((3,4), (1,2))
# plt.imshow(np.load('GAN_8_dim_no_sigmoid_z_optimisation_one_lambda_0.01953125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# plt.yticks([])
# plt.xticks([])
# plt.xlabel('lambda={:.2E}'.format(0.01953125))


# # small subplot 4
# plt.subplot2grid((3,4), (1,3))
# plt.imshow(np.load('GAN_8_dim_no_sigmoid_z_optimisation_one_lambda_0.0048828125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# plt.yticks([])
# plt.xticks([])
# plt.xlabel('lambda={:.2E}'.format(0.0048828125))



# # fit subplots and save fig
# fig.tight_layout()
# fig.set_size_inches(w=11,h=6)
# fig_name = 'gan_z_opt_one.png'
# fig.savefig(fig_name, bbox_inches='tight')

# #%%
# # Plot figure with subplots of different sizes
# fig = plt.figure(1)
# # set up subplot grid
# gridspec.GridSpec(3,4)

# # large subplot
# plt.subplot2grid((3,4), (0,0), colspan=2, rowspan=2)
# plot=gan_x_soft
# plt.scatter(plot[:,0], plot[:,2], c=plot[:,1],s=10)
# plt.colorbar()
# plt.hlines(28*28*0.1**2, np.min(plot[:,0]), np.max(plot[:,0]))
# plt.xscale('log')
# plt.xlabel('lambda')
# plt.ylabel('Data discrepancy')
# plt.ylim(0,30)


# # small subplot 1
# plt.subplot2grid((3,4), (0,2))
# plt.imshow(np.load('GAN_8_dim_no_sigmoid_x_soft_onelambda_2.5_mu_2.5.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# plt.yticks([])
# plt.xticks([])
# plt.xlabel('lambda={:.2E},\n $\mu$={:.2E}'.format(2.5,2.5))

# # small subplot 2
# plt.subplot2grid((3,4), (0,3))
# plt.imshow(np.load('GAN_8_dim_no_sigmoid_x_soft_onelambda_0.15625_mu_0.3125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# plt.yticks([])
# plt.xticks([])
# plt.xlabel('lambda={:.2E},\n $\mu$={:.2E}'.format(0.15625,0.3125))


# # small subplot 3
# plt.subplot2grid((3,4), (1,2))
# plt.imshow(np.load('GAN_8_dim_no_sigmoid_x_soft_onelambda_0.01953125_mu_0.078125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# plt.yticks([])
# plt.xticks([])
# plt.xlabel('lambda={:.2E},\n  $\mu$={:.2E}'.format(0.01953125,0.078125))


# # small subplot 4
# plt.subplot2grid((3,4), (1,3))
# plt.imshow(np.load('GAN_8_dim_no_sigmoid_x_soft_onelambda_0.0048828125_mu_0.01953125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# plt.yticks([])
# plt.xticks([])
# plt.xlabel('lambda={:.2E},\n  $\mu$={:.2E}'.format(0.0048828125,0.01953125))



# # fit subplots and save fig
# fig.tight_layout()
# fig.set_size_inches(w=11,h=6)
# fig_name = 'gan_x_soft_one.png'
# fig.savefig(fig_name, bbox_inches='tight')

# #%%

# # Plot figure with subplots of different sizes
# fig = plt.figure(1)
# # set up subplot grid
# gridspec.GridSpec(3,4)

# # large subplot
# plt.subplot2grid((3,4), (0,0), colspan=2, rowspan=2)
# plot=gan_z_sparse
# plt.scatter(plot[:,0], plot[:,2], c=plot[:,1],s=10)
# plt.colorbar()
# plt.hlines(28*28*0.1**2, np.min(plot[:,0]), np.max(plot[:,0]))
# plt.xscale('log')
# plt.xlabel('lambda')
# plt.ylabel('Data discrepancy')
# plt.ylim(0,30)


# # small subplot 1
# plt.subplot2grid((3,4), (0,2))
# plt.imshow(np.load('GAN_8_dim_no_sigmoid_z_sparse_one_lambda_2.5_mu_2.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# plt.yticks([])
# plt.xticks([])
# plt.xlabel('lambda={:.2E},\n  $\mu$={:.2E}'.format(2.5,2.5))

# # small subplot 2
# plt.subplot2grid((3,4), (0,3))
# plt.imshow(np.load('GAN_8_dim_no_sigmoid_z_sparse_one_lambda_0.15625_mu_0.25.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# plt.yticks([])
# plt.xticks([])
# plt.xlabel('lambda={:.2E},\n  $\mu$={:.2E}'.format(0.15625,0.25))


# # small subplot 3
# plt.subplot2grid((3,4), (1,2))
# plt.imshow(np.load('GAN_8_dim_no_sigmoid_z_sparse_one_lambda_0.01953125_mu_0.03125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# plt.yticks([])
# plt.xticks([])
# plt.xlabel('lambda={:.2E},\n $\mu$={:.2E}'.format(0.01953125,0.03125))


# # small subplot 4
# plt.subplot2grid((3,4), (1,3))
# plt.imshow(np.load('GAN_8_dim_no_sigmoid_z_sparse_one_lambda_0.0048828125_mu_0.015625.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# plt.yticks([])
# plt.xticks([])
# plt.xlabel('lambda={:.2E}, \n $\mu$={:.2E}'.format(0.0048828125,0.015625))



# # fit subplots and save fig
# fig.tight_layout()
# fig.set_size_inches(w=11,h=6)
# fig_name = 'gan_z_sparse_one.png'
# fig.savefig(fig_name, bbox_inches='tight')


#%%  NEW plots - 16/12/2020

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SMALL_SIZE = 8
MEDIUM_SIZE = 8
BIGGER_SIZE = 8

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
aim=np.load('aim_mnist_convolution_one.npy')


#%%
#%%

ae_x_tik=np.load('AE_8_dim_no_sigmoid_x_tik_oneloss.npy')
ae_x_soft=np.load('AE_8_dim_no_sigmoid_x_soft_onelambda_mu_loss.npy')
ae_z_sparse=np.load('AE_8_dim_no_sigmoid_z_sparse_one_lambda_mu_loss.npy')
ae_z_opt=np.load('AE_8_dim_no_sigmoid_z_optimisation_one__loss.npy')



# Plot figure with subplots of different sizes
fig = plt.figure(1)
# set up subplot grid
gridspec.GridSpec(5,5)

# large subplot
plt.subplot2grid((5,5), (0,0), colspan=5, rowspan=3)
plot=ae_z_opt
plt.scatter(plot[:,0], plot[:,1], s=10)
plt.hlines(28*28*0.1**2, np.min(plot[:,0]), np.max(plot[:,0]))
plt.xscale('log')
plt.xlabel('$\lambda$')
plt.ylabel('Data discrepancy')
plt.ylim(0,20)
#.colorbar()

# small subplot 1
plt.subplot2grid((5,5), (4,0))
plt.imshow(aim, cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('Ground truth')
# small subplot 1
plt.subplot2grid((5,5), (4,1))
plt.imshow(np.load('AE_8_dim_no_sigmoid_z_optimisation_one_lambda_2.5.npy'), cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('$\lambda$={:.2E}'.format(2.5))

# small subplot 2
plt.subplot2grid((5,5), (4,2))
plt.imshow(np.load('AE_8_dim_no_sigmoid_z_optimisation_one_lambda_0.15625.npy'), cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('$\lambda$={:.2E}'.format(0.15625))


# small subplot 3
plt.subplot2grid((5,5), (4,3))
plt.imshow(np.load('AE_8_dim_no_sigmoid_z_optimisation_one_lambda_0.01953125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('$\lambda$={:.2E}'.format(0.01953125))


# small subplot 4
plt.subplot2grid((5,5), (4,4))
plt.imshow(np.load('AE_8_dim_no_sigmoid_z_optimisation_one_lambda_0.0048828125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('$\lambda$={:.2E}'.format(0.0048828125))



# fit subplots and save fig

fig_name = 'ae_z_opt_one.png'
fig.savefig(fig_name, bbox_inches='tight')

#%%

# Plot figure with subplots of different sizes
fig = plt.figure(1)
# set up subplot grid
gridspec.GridSpec(5,5)

# large subplot
plt.subplot2grid((5,5), (0,0), colspan=5, rowspan=3)
plot=ae_x_tik
plt.scatter(plot[:,0], plot[:,1], s=10)
plt.hlines(28*28*0.1**2, np.min(plot[:,0]), np.max(plot[:,0]))
plt.xscale('log')
plt.xlabel('$\lambda$')
plt.ylabel('Data discrepancy')
plt.ylim(0,20)
#plt.colorbar()

# small subplot 1
plt.subplot2grid((5,5), (4,0))
plt.imshow(aim, cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('Ground truth')

# small subplot 1
plt.subplot2grid((5,5), (4,1))
plt.imshow(np.load('AE_8_dim_no_sigmoid_x_tik_one_lambda_2.5.npy'), cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('$\lambda$={:.2E}'.format(2.5))

# small subplot 2
plt.subplot2grid((5,5), (4,2))
plt.imshow(np.load('AE_8_dim_no_sigmoid_x_tik_one_lambda_0.15625.npy'), cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('$\lambda$={:.2E}'.format(0.15625))


# small subplot 3
plt.subplot2grid((5,5), (4,3))
plt.imshow(np.load('AE_8_dim_no_sigmoid_x_tik_one_lambda_0.01953125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('$\lambda$={:.2E}'.format(0.01953125))


# small subplot 4
plt.subplot2grid((5,5), (4,4))
plt.imshow(np.load('AE_8_dim_no_sigmoid_x_tik_one_lambda_0.0048828125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('$\lambda$={:.2E}'.format(0.0048828125))



# fit subplots and save fig

fig_name = 'ae_x_tik_one.png'
fig.savefig(fig_name, bbox_inches='tight')

#%%

# Plot figure with subplots of different sizes
fig = plt.figure(1)
# set up subplot grid
gridspec.GridSpec(5,5)

# large subplot
plt.subplot2grid((5,5), (0,0), colspan=5, rowspan=3)
plot=ae_x_soft
plt.scatter(plot[:,0], plot[:,2], c=plot[:,1],s=10)
plt.colorbar()
plt.hlines(28*28*0.1**2, np.min(plot[:,0]), np.max(plot[:,0]))
plt.xscale('log')
plt.xlabel('$\lambda$')
plt.ylabel('Data discrepancy')
plt.ylim(0,20)

# small subplot 1
plt.subplot2grid((5,5), (4,0))
plt.imshow(aim, cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('Ground truth')

# small subplot 1
plt.subplot2grid((5,5), (4,1))
plt.imshow(np.load('AE_8_dim_no_sigmoid_x_soft_onelambda_2.5_mu_2.5.npy'), cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('$\lambda$={:.2E},\n $\mu$={:.2E}'.format(2.5,2.5))

# small subplot 2
plt.subplot2grid((5,5), (4,2))
plt.imshow(np.load('AE_8_dim_no_sigmoid_x_soft_onelambda_0.15625_mu_0.3125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('$\lambda$={:.2E},\n $\mu$={:.2E}'.format(0.15625,0.3125))


# small subplot 3
plt.subplot2grid((5,5), (4,3))
plt.imshow(np.load('AE_8_dim_no_sigmoid_x_soft_onelambda_0.01953125_mu_0.078125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('$\lambda$={:.2E},\n  $\mu$={:.2E}'.format(0.01953125,0.078125))


# small subplot 4
plt.subplot2grid((5,5), (4,4))
plt.imshow(np.load('AE_8_dim_no_sigmoid_x_soft_onelambda_0.0048828125_mu_0.01953125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('$\lambda$={:.2E},\n  $\mu$={:.2E}'.format(0.0048828125,0.01953125))



# fit subplots and save fig

fig_name = 'ae_x_soft_one.png'
fig.savefig(fig_name, bbox_inches='tight')

#%%

# Plot figure with subplots of different sizes
fig = plt.figure(1)
# set up subplot grid
gridspec.GridSpec(5,5)

# large subplot
plt.subplot2grid((5,5), (0,0), colspan=5, rowspan=3)
plot=ae_z_sparse
plt.scatter(plot[:,0], plot[:,2], c=plot[:,1],s=10)
plt.colorbar()
plt.hlines(28*28*0.1**2, np.min(plot[:,0]), np.max(plot[:,0]))
plt.xscale('log')
plt.xlabel('$\lambda$')
plt.ylabel('Data discrepancy')
plt.ylim(0,20)

# small subplot 1
plt.subplot2grid((5,5), (4,0))
plt.imshow(aim, cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('Ground truth')

# small subplot 1
plt.subplot2grid((5,5), (4,1))
plt.imshow(np.load('AE_8_dim_no_sigmoid_z_sparse_one_lambda_2.5_mu_2.npy'), cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('$\lambda$={:.2E},\n  $\mu$={:.2E}'.format(2.5,2.5))

# small subplot 2
plt.subplot2grid((5,5), (4,2))
plt.imshow(np.load('AE_8_dim_no_sigmoid_z_sparse_one_lambda_0.15625_mu_0.25.npy'), cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('$\lambda$={:.2E},\n  $\mu$={:.2E}'.format(0.15625,0.25))


# small subplot 3
plt.subplot2grid((5,5), (4,3))
plt.imshow(np.load('AE_8_dim_no_sigmoid_z_sparse_one_lambda_0.01953125_mu_0.03125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('$\lambda$={:.2E},\n $\mu$={:.2E}'.format(0.01953125,0.03125))


# small subplot 4
plt.subplot2grid((5,5), (4,4))
plt.imshow(np.load('AE_8_dim_no_sigmoid_z_sparse_one_lambda_0.0048828125_mu_0.015625.npy'), cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('$\lambda$={:.2E}, \n $\mu$={:.2E}'.format(0.0048828125,0.015625))



# fit subplots and save fig

fig_name = 'ae_z_sparse_one.png'
fig.savefig(fig_name, bbox_inches='tight')



#%%

vae_x_soft=np.load('VAE_8_dim_no_sigmoid_x_soft_onelambda_mu_loss.npy')
vae_z_sparse=np.load('VAE_8_dim_no_sigmoid_z_sparse_one_lambda_mu_loss.npy')
vae_z_opt=np.load('VAE_8_dim_no_sigmoid_z_optimisation_one__loss.npy')



# Plot figure with subplots of different sizes
fig = plt.figure(1)
# set up subplot grid
gridspec.GridSpec(5,5)

# large subplot
plt.subplot2grid((5,5), (0,0), colspan=5, rowspan=3)
plot=vae_z_opt
plt.scatter(plot[:,0], plot[:,1], s=10)
plt.hlines(28*28*0.1**2, np.min(plot[:,0]), np.max(plot[:,0]))
plt.xscale('log')
plt.xlabel('$\lambda$')
plt.ylabel('Data discrepancy')
plt.ylim(0,20)
#.colorbar()

# small subplot 1
plt.subplot2grid((5,5), (4,0))
plt.imshow(aim, cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('Ground truth')


# small subplot 1
plt.subplot2grid((5,5), (4,1))
plt.imshow(np.load('VAE_8_dim_no_sigmoid_z_optimisation_one_lambda_2.5.npy'), cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('$\lambda$={:.2E}'.format(2.5))

# small subplot 2
plt.subplot2grid((5,5), (4,2))
plt.imshow(np.load('VAE_8_dim_no_sigmoid_z_optimisation_one_lambda_0.15625.npy'), cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('$\lambda$={:.2E}'.format(0.15625))


# small subplot 3
plt.subplot2grid((5,5), (4,3))
plt.imshow(np.load('VAE_8_dim_no_sigmoid_z_optimisation_one_lambda_0.01953125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('$\lambda$={:.2E}'.format(0.01953125))


# small subplot 4
plt.subplot2grid((5,5), (4,4))
plt.imshow(np.load('VAE_8_dim_no_sigmoid_z_optimisation_one_lambda_0.0048828125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('$\lambda$={:.2E}'.format(0.0048828125))



# fit subplots and save fig

fig_name = 'vae_z_opt_one.png'
fig.savefig(fig_name, bbox_inches='tight')

#%%
# Plot figure with subplots of different sizes
fig = plt.figure(1)
# set up subplot grid
gridspec.GridSpec(5,5)

# large subplot
plt.subplot2grid((5,5), (0,0), colspan=5, rowspan=3)
plot=vae_x_soft
plt.scatter(plot[:,0], plot[:,2], c=plot[:,1],s=10)
plt.colorbar()
plt.hlines(28*28*0.1**2, np.min(plot[:,0]), np.max(plot[:,0]))
plt.xscale('log')
plt.xlabel('$\lambda$')
plt.ylabel('Data discrepancy')
plt.ylim(0,20)

# small subplot 1
plt.subplot2grid((5,5), (4,0))
plt.imshow(aim, cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('Ground truth')


# small subplot 1
plt.subplot2grid((5,5), (4,1))
plt.imshow(np.load('VAE_8_dim_no_sigmoid_x_soft_onelambda_2.5_mu_2.5.npy'), cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('$\lambda$={:.2E},\n $\mu$={:.2E}'.format(2.5,2.5))

# small subplot 2
plt.subplot2grid((5,5), (4,2))
plt.imshow(np.load('VAE_8_dim_no_sigmoid_x_soft_onelambda_0.15625_mu_0.3125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('$\lambda$={:.2E},\n $\mu$={:.2E}'.format(0.15625,0.3125))


# small subplot 3
plt.subplot2grid((5,5), (4,3))
plt.imshow(np.load('VAE_8_dim_no_sigmoid_x_soft_onelambda_0.01953125_mu_0.078125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('$\lambda$={:.2E},\n  $\mu$={:.2E}'.format(0.01953125,0.078125))


# small subplot 4
plt.subplot2grid((5,5), (4,4))
plt.imshow(np.load('VAE_8_dim_no_sigmoid_x_soft_onelambda_0.0048828125_mu_0.01953125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('$\lambda$={:.2E},\n  $\mu$={:.2E}'.format(0.0048828125,0.01953125))



# fit subplots and save fig
fig_name = 'vae_x_soft_one.png'
fig.savefig(fig_name, bbox_inches='tight')

#%%

# Plot figure with subplots of different sizes
fig = plt.figure(1)
# set up subplot grid
gridspec.GridSpec(5,5)

# large subplot
plt.subplot2grid((5,5), (0,0), colspan=5, rowspan=3)
plot=vae_z_sparse
plt.scatter(plot[:,0], plot[:,2], c=plot[:,1],s=10)
plt.colorbar()
plt.hlines(28*28*0.1**2, np.min(plot[:,0]), np.max(plot[:,0]))
plt.xscale('log')
plt.xlabel('$\lambda$')
plt.ylabel('Data discrepancy')
plt.ylim(0,20)

# small subplot 1
plt.subplot2grid((5,5), (4,0))
plt.imshow(aim, cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('Ground truth')


# small subplot 1
plt.subplot2grid((5,5), (4,1))
plt.imshow(np.load('VAE_8_dim_no_sigmoid_z_sparse_one_lambda_2.5_mu_2.npy'), cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('$\lambda$={:.2E},\n  $\mu$={:.2E}'.format(2.5,2.5))

# small subplot 2
plt.subplot2grid((5,5), (4,2))
plt.imshow(np.load('VAE_8_dim_no_sigmoid_z_sparse_one_lambda_0.15625_mu_0.25.npy'), cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('$\lambda$={:.2E},\n  $\mu$={:.2E}'.format(0.15625,0.25))


# small subplot 3
plt.subplot2grid((5,5), (4,3))
plt.imshow(np.load('VAE_8_dim_no_sigmoid_z_sparse_one_lambda_0.01953125_mu_0.03125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('$\lambda$={:.2E},\n $\mu$={:.2E}'.format(0.01953125,0.03125))


# small subplot 4
plt.subplot2grid((5,5), (4,4))
plt.imshow(np.load('VAE_8_dim_no_sigmoid_z_sparse_one_lambda_0.0048828125_mu_0.015625.npy'), cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('$\lambda$={:.2E}, \n $\mu$={:.2E}'.format(0.0048828125,0.015625))



# fit subplots and save fig

fig_name = 'vae_z_sparse_one.png'
fig.savefig(fig_name, bbox_inches='tight')




#%%



gan_x_soft=np.load('GAN_8_dim_no_sigmoid_x_soft_onelambda_mu_loss.npy')
gan_z_sparse=np.load('GAN_8_dim_no_sigmoid_z_sparse_one_lambda_mu_loss.npy')
gan_z_opt=np.load('GAN_8_dim_no_sigmoid_z_optimisation_one__loss.npy')



# Plot figure with subplots of different sizes
fig = plt.figure(1)
# set up subplot grid
gridspec.GridSpec(4,3)

# large subplot
plt.subplot2grid((5,5), (0,0), colspan=5, rowspan=3)
plot=gan_z_opt
plt.scatter(plot[:,0], plot[:,1], s=10)
plt.hlines(28*28*0.1**2, np.min(plot[:,0]), np.max(plot[:,0]))
plt.xscale('log')
plt.xlabel('$\lambda$')
plt.ylabel('Data discrepancy')
plt.ylim(0,20)
#.colorbar()


# small subplot 1
plt.subplot2grid((5,5), (4,0))
plt.imshow(aim, cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('Ground truth')


# small subplot 1
plt.subplot2grid((5,5), (4,1))
plt.imshow(np.load('GAN_8_dim_no_sigmoid_z_optimisation_one_lambda_2.5.npy'), cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('$\lambda$={:.2E}'.format(2.5))

# small subplot 2
plt.subplot2grid((5,5), (4,2))
plt.imshow(np.load('GAN_8_dim_no_sigmoid_z_optimisation_one_lambda_0.15625.npy'), cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('$\lambda$={:.2E}'.format(0.15625))


# small subplot 3
plt.subplot2grid((5,5), (4,3))
plt.imshow(np.load('GAN_8_dim_no_sigmoid_z_optimisation_one_lambda_0.01953125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('$\lambda$={:.2E}'.format(0.01953125))


# small subplot 4
plt.subplot2grid((5,5), (4,4))
plt.imshow(np.load('GAN_8_dim_no_sigmoid_z_optimisation_one_lambda_0.0048828125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('$\lambda$={:.2E}'.format(0.0048828125))



# fit subplots and save fig
#fig.tight_layout()

fig_name = 'gan_z_opt_one.png'
fig.savefig(fig_name, bbox_inches='tight')

#%%
# Plot figure with subplots of different sizes
fig = plt.figure(1)
# set up subplot grid
gridspec.GridSpec(5,5)

# large subplot
plt.subplot2grid((5,5), (0,0), colspan=5, rowspan=3)
plot=gan_x_soft
plt.scatter(plot[:,0], plot[:,2], c=plot[:,1],s=10)
plt.colorbar()
plt.hlines(28*28*0.1**2, np.min(plot[:,0]), np.max(plot[:,0]))
plt.xscale('log')
plt.xlabel('$\lambda$')
plt.ylabel('Data discrepancy')
plt.ylim(0,20)

# small subplot 1
plt.subplot2grid((5,5), (4,0))
plt.imshow(aim, cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('Ground truth')


# small subplot 1
plt.subplot2grid((5,5), (4,1))
plt.imshow(np.load('GAN_8_dim_no_sigmoid_x_soft_onelambda_2.5_mu_2.5.npy'), cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('$\lambda$={:.2E},\n $\mu$={:.2E}'.format(2.5,2.5))

# small subplot 2
plt.subplot2grid((5,5), (4,2))
plt.imshow(np.load('GAN_8_dim_no_sigmoid_x_soft_onelambda_0.15625_mu_0.3125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('$\lambda$={:.2E},\n $\mu$={:.2E}'.format(0.15625,0.3125))


# small subplot 3
plt.subplot2grid((5,5), (4,3))
plt.imshow(np.load('GAN_8_dim_no_sigmoid_x_soft_onelambda_0.01953125_mu_0.078125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('$\lambda$={:.2E},\n  $\mu$={:.2E}'.format(0.01953125,0.078125))


# small subplot 4
plt.subplot2grid((5,5), (4,4))
plt.imshow(np.load('GAN_8_dim_no_sigmoid_x_soft_onelambda_0.0048828125_mu_0.01953125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('$\lambda$={:.2E},\n  $\mu$={:.2E}'.format(0.0048828125,0.01953125))



# fit subplots and save fig
#fig.tight_layout()

fig_name = 'gan_x_soft_one.png'
fig.savefig(fig_name, bbox_inches='tight')

#%%

# Plot figure with subplots of different sizes
fig = plt.figure(1)
# set up subplot grid
gridspec.GridSpec(5,5)

# large subplot
plt.subplot2grid((5,5), (0,0), colspan=5, rowspan=3)
plot=gan_z_sparse
plt.scatter(plot[:,0], plot[:,2], c=plot[:,1],s=10)
plt.colorbar()
plt.hlines(28*28*0.1**2, np.min(plot[:,0]), np.max(plot[:,0]))
plt.xscale('log')
plt.xlabel('$\lambda$')
plt.ylabel('Data discrepancy')
plt.ylim(0,20)

# small subplot 1
plt.subplot2grid((5,5), (4,0))
plt.imshow(aim, cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('Ground truth')

# small subplot 1
plt.subplot2grid((5,5), (4,1))
plt.imshow(np.load('GAN_8_dim_no_sigmoid_z_sparse_one_lambda_2.5_mu_2.npy'), cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('$\lambda$={:.2E},\n  $\mu$={:.2E}'.format(2.5,2.5))

# small subplot 2
plt.subplot2grid((5,5), (4,2))
plt.imshow(np.load('GAN_8_dim_no_sigmoid_z_sparse_one_lambda_0.15625_mu_0.25.npy'), cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('$\lambda$={:.2E},\n  $\mu$={:.2E}'.format(0.15625,0.25))


# small subplot 3
plt.subplot2grid((5,5), (4,3))
plt.imshow(np.load('GAN_8_dim_no_sigmoid_z_sparse_one_lambda_0.01953125_mu_0.03125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('$\lambda$={:.2E},\n $\mu$={:.2E}'.format(0.01953125,0.03125))


# small subplot 4
plt.subplot2grid((5,5), (4,4))
plt.imshow(np.load('GAN_8_dim_no_sigmoid_z_sparse_one_lambda_0.0048828125_mu_0.015625.npy'), cmap='gray', vmin=-.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
plt.xlabel('$\lambda$={:.2E}, \n $\mu$={:.2E}'.format(0.0048828125,0.015625))



# fit subplots and save fig
#fig.tight_layout()
#fig.set_size_inches(w=11,h=6)
fig_name = 'gan_z_sparse_one.png'
fig.savefig(fig_name, bbox_inches='tight')

#%%

# #%%


# ae_x_tik=np.load('AE_8_dim_no_sigmoid_x_tik_oneloss.npy')
# ae_x_soft=np.load('AE_8_dim_no_sigmoid_x_soft_onelambda_mu_loss.npy')
# ae_z_sparse=np.load('AE_8_dim_no_sigmoid_z_sparse_one_lambda_mu_loss.npy')
# ae_z_opt=np.load('AE_8_dim_no_sigmoid_z_optimisation_one__loss.npy')


# fig, axes=plt.subplots(5,4, figsize=(40,50))

# plot=ae_z_opt
# axes[0,0].scatter(plot[:,0], plot[:,1], s=10)
# axes[0,0].hlines(28*28*0.1**2, np.min(plot[:,0]), np.max(plot[:,0]))
# axes[0,0].set_xscale('log')
# axes[0,0].set_xlabel('lambda')
# axes[0,0].set_ylabel('Data discrepancy')
# axes[0,0].set_ylim(0,30)

# plot=ae_x_soft
# im=axes[0,1].scatter(plot[:,0], plot[:,2], c=plot[:,1],s=10)
# fig.colorbar(im,cax=axes[0,1])
# axes[0,1].hlines(28*28*0.1**2, np.min(plot[:,0]), np.max(plot[:,0]))
# axes[0,1].set_xscale('log')
# axes[0,1].set_xlabel('lambda')
# axes[0,1].set_ylabel('Data discrepancy')
# axes[0,1].set_ylim(0,30)

# plot=ae_z_sparse
# im=axes[0,2].scatter(plot[:,0], plot[:,2], c=plot[:,1], s=10)
# fig.colorbar(im,cax=axes[0,2])
# axes[0,2].hlines(28*28*0.1**2, np.min(plot[:,0]), np.max(plot[:,0]))
# axes[0,2].set_xscale('log')
# axes[0,2].set_xlabel('lambda')
# axes[0,2].set_ylabel('Data discrepancy')
# axes[0,2].set_ylim(0,30)

# plot=ae_x_tik
# axes[0,3].scatter(plot[:,0], plot[:,1], s=10)
# axes[0,3].hlines(28*28*0.1**2, np.min(plot[:,0]), np.max(plot[:,0]))
# axes[0,3].set_xscale('log')
# axes[0,3].set_xlabel('lambda')
# axes[0,3].set_ylabel('Data discrepancy')
# axes[0,3].set_ylim(0,30)

# axes[1,0].imshow(np.load('AE_8_dim_no_sigmoid_z_optimisation_one_lambda_2.5.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# axes[1,0].set_yticklabels([])
# axes[1,0].set_xticklabels([])
# axes[1,0].set_xlabel('lambda={:.2E}'.format(2.5))
# axes[2,0].imshow(np.load('AE_8_dim_no_sigmoid_z_optimisation_one_lambda_0.15625.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# axes[2,0].set_yticklabels([])
# axes[2,0].set_xticklabels([])
# axes[2,0].set_xlabel('lambda={:.2E}'.format(0.15625))
# axes[3,0].imshow(np.load('AE_8_dim_no_sigmoid_z_optimisation_one_lambda_0.01953125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# axes[3,0].set_yticklabels([])
# axes[3,0].set_xticklabels([])
# axes[3,0].set_xlabel('lambda={:.2E}'.format(0.01953125))
# axes[4,0].imshow(np.load('AE_8_dim_no_sigmoid_z_optimisation_one_lambda_0.0048828125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# axes[4,0].set_yticklabels([])
# axes[4,0].set_xticklabels([])
# axes[4,0].set_xlabel('lambda={:.2E}'.format(0.0048828125))

# axes[1,1].imshow(np.load('AE_8_dim_no_sigmoid_x_soft_onelambda_2.5_mu_2.5.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# axes[1,1].set_yticklabels([])
# axes[1,1].set_xticklabels([])
# axes[1,1].set_xlabel('lambda={:.2E}, $\mu$={:.2E}'.format(2.5,2.5))
# axes[2,1].imshow(np.load('AE_8_dim_no_sigmoid_x_soft_onelambda_0.15625_mu_0.3125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# axes[2,1].set_yticklabels([])
# axes[2,1].set_xticklabels([])
# axes[2,1].set_xlabel('lambda={:.2E}, $\mu$={:.2E}'.format(0.15625,0.3125))
# axes[3,1].imshow(np.load('AE_8_dim_no_sigmoid_x_soft_onelambda_0.01953125_mu_0.078125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# axes[3,1].set_yticklabels([])
# axes[3,1].set_xticklabels([])
# axes[3,1].set_xlabel('lambda={:.2E}, $\mu$={:.2E}'.format(0.01953125,0.078125))
# axes[4,1].imshow(np.load('AE_8_dim_no_sigmoid_x_soft_onelambda_0.0048828125_mu_0.01953125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# axes[4,1].set_yticklabels([])
# axes[4,1].set_xticklabels([])
# axes[4,1].set_xlabel('lambda={:.2E}, $\mu$={:.2E}'.format(0.0048828125,0.01953125))

# axes[1,2].imshow(np.load('AE_8_dim_no_sigmoid_z_sparse_one_lambda_2.5_mu_2.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# axes[1,2].set_yticklabels([])
# axes[1,2].set_xticklabels([])
# axes[1,2].set_xlabel('lambda={:.2E}, $\mu$={:.2E}'.format(2.5,2))
# axes[2,2].imshow(np.load('AE_8_dim_no_sigmoid_z_sparse_one_lambda_0.15625_mu_0.25.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# axes[2,2].set_yticklabels([])
# axes[2,2].set_xticklabels([])
# axes[2,2].set_xlabel('lambda={:.2E}, $\mu$={:.2E}'.format(0.15625,0.25))
# axes[3,2].imshow(np.load('AE_8_dim_no_sigmoid_z_sparse_one_lambda_0.01953125_mu_0.03125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# axes[3,2].set_yticklabels([])
# axes[3,2].set_xticklabels([])
# axes[3,2].set_xlabel('lambda={:.2E}, $\mu$={:.2E}'.format(0.01953125,0.03125))
# axes[4,2].imshow(np.load('AE_8_dim_no_sigmoid_z_sparse_one_lambda_0.0048828125_mu_0.015625.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# axes[4,2].set_yticklabels([])
# axes[4,2].set_xticklabels([])
# axes[4,2].set_xlabel('lambda={:.2E}, $\mu$={:.2E}'.format(0.0048828125,0.015625))





# axes[1,3].imshow(np.load('AE_8_dim_no_sigmoid_x_tik_one_lambda_2.5.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# axes[1,3].set_yticklabels([])
# axes[1,3].set_xticklabels([])
# axes[1,3].set_xlabel('lambda={:.2E}'.format(2.5))
# axes[2,3].imshow(np.load('AE_8_dim_no_sigmoid_x_tik_one_lambda_0.15625.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# axes[2,3].set_yticklabels([])
# axes[2,3].set_xticklabels([])
# axes[2,3].set_xlabel('lambda={:.2E}'.format(0.15625))
# axes[3,3].imshow(np.load('AE_8_dim_no_sigmoid_x_tik_one_lambda_0.01953125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# axes[3,3].set_yticklabels([])
# axes[3,3].set_xticklabels([])
# axes[3,3].set_xlabel('lambda={:.2E}'.format(0.01953125))
# axes[4,3].imshow(np.load('AE_8_dim_no_sigmoid_x_tik_one_lambda_0.0048828125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# axes[4,3].set_yticklabels([])
# axes[4,3].set_xticklabels([])
# axes[4,3].set_xlabel('lambda={:.2E}'.format(0.0048828125))

# plt.savefig('AE_mnist_8_one.png')


# #%%



# vae_x_tik=np.load('AE_8_dim_no_sigmoid_x_tik_oneloss.npy')
# vae_x_soft=np.load('VAE_8_dim_no_sigmoid_x_soft_onelambda_mu_loss.npy')
# vae_z_sparse=np.load('VAE_8_dim_no_sigmoid_z_sparse_one_lambda_mu_loss.npy')
# vae_z_opt=np.load('VAE_8_dim_no_sigmoid_z_optimisation_one__loss.npy')


# fig, axes=plt.subplots(5,4, figsize=(40,50))

# plot=vae_z_opt
# axes[0,0].scatter(plot[:,0], plot[:,1])
# axes[0,0].hlines(28*28*0.1**2, np.min(plot[:,0]), np.max(plot[:,0]))
# axes[0,0].set_xscale('log')
# axes[0,0].set_xlabel('lambda')
# axes[0,0].set_ylabel('Data discrepancy')
# axes[0,0].set_ylim(0,30)

# plot=vae_x_soft
# im=axes[0,1].scatter(plot[:,0], plot[:,2], c=plot[:,1])
# #plt.colorbar(im,cax=axes[0,1])
# axes[0,1].hlines(28*28*0.1**2, np.min(plot[:,0]), np.max(plot[:,0]))
# axes[0,1].set_xscale('log')
# axes[0,1].set_xlabel('lambda')
# axes[0,1].set_ylabel('Data discrepancy')
# axes[0,1].set_ylim(0,30)

# plot=vae_z_sparse
# im=axes[0,2].scatter(plot[:,0], plot[:,2], c=plot[:,1])
# #plt.colorbar(im,cax=axes[0,2])
# axes[0,2].hlines(28*28*0.1**2, np.min(plot[:,0]), np.max(plot[:,0]))
# axes[0,2].set_xscale('log')
# axes[0,2].set_xlabel('lambda')
# axes[0,2].set_ylabel('Data discrepancy')
# axes[0,2].set_ylim(0,30)

# plot=vae_x_tik
# axes[0,3].scatter(plot[:,0], plot[:,1])
# axes[0,3].hlines(28*28*0.1**2, np.min(plot[:,0]), np.max(plot[:,0]))
# axes[0,3].set_xscale('log')
# axes[0,3].set_xlabel('lambda')
# axes[0,3].set_ylabel('Data discrepancy')
# axes[0,3].set_ylim(0,30)

# axes[1,0].imshow(np.load('VAE_8_dim_no_sigmoid_z_optimisation_one_lambda_2.5.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# axes[1,0].set_yticklabels([])
# axes[1,0].set_xticklabels([])
# axes[1,0].set_xlabel('lambda={:.2E}'.format(2.5))
# axes[2,0].imshow(np.load('VAE_8_dim_no_sigmoid_z_optimisation_one_lambda_0.15625.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# axes[2,0].set_yticklabels([])
# axes[2,0].set_xticklabels([])
# axes[2,0].set_xlabel('lambda={:.2E}'.format(0.15625))
# axes[3,0].imshow(np.load('VAE_8_dim_no_sigmoid_z_optimisation_one_lambda_0.01953125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# axes[3,0].set_yticklabels([])
# axes[3,0].set_xticklabels([])
# axes[3,0].set_xlabel('lambda={:.2E}'.format(0.01953125))
# axes[4,0].imshow(np.load('VAE_8_dim_no_sigmoid_z_optimisation_one_lambda_0.0048828125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# axes[4,0].set_yticklabels([])
# axes[4,0].set_xticklabels([])
# axes[4,0].set_xlabel('lambda={:.2E}'.format(0.0048828125))

# axes[1,1].imshow(np.load('VAE_8_dim_no_sigmoid_x_soft_onelambda_2.5_mu_2.5.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# axes[1,1].set_yticklabels([])
# axes[1,1].set_xticklabels([])
# axes[1,1].set_xlabel('lambda={:.2E}, $\mu$={:.2E}'.format(2.5,2.5))
# axes[2,1].imshow(np.load('VAE_8_dim_no_sigmoid_x_soft_onelambda_0.15625_mu_0.3125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# axes[2,1].set_yticklabels([])
# axes[2,1].set_xticklabels([])
# axes[2,1].set_xlabel('lambda={:.2E}, $\mu$={:.2E}'.format(0.15625,0.3125))
# axes[3,1].imshow(np.load('VAE_8_dim_no_sigmoid_x_soft_onelambda_0.01953125_mu_0.078125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# axes[3,1].set_yticklabels([])
# axes[3,1].set_xticklabels([])
# axes[3,1].set_xlabel('lambda={:.2E}, $\mu$={:.2E}'.format(0.01953125,0.078125))
# axes[4,1].imshow(np.load('VAE_8_dim_no_sigmoid_x_soft_onelambda_0.0048828125_mu_0.01953125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# axes[4,1].set_yticklabels([])
# axes[4,1].set_xticklabels([])
# axes[4,1].set_xlabel('lambda={:.2E}, $\mu$={:.2E}'.format(0.0048828125,0.01953125))

# axes[1,2].imshow(np.load('VAE_8_dim_no_sigmoid_z_sparse_one_lambda_2.5_mu_2.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# axes[1,2].set_yticklabels([])
# axes[1,2].set_xticklabels([])
# axes[1,2].set_xlabel('lambda={:.2E}, $\mu$={:.2E}'.format(2.5,2))
# axes[2,2].imshow(np.load('VAE_8_dim_no_sigmoid_z_sparse_one_lambda_0.15625_mu_0.25.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# axes[2,2].set_yticklabels([])
# axes[2,2].set_xticklabels([])
# axes[2,2].set_xlabel('lambda={:.2E}, $\mu$={:.2E}'.format(0.15625,0.25))
# axes[3,2].imshow(np.load('VAE_8_dim_no_sigmoid_z_sparse_one_lambda_0.01953125_mu_0.03125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# axes[3,2].set_yticklabels([])
# axes[3,2].set_xticklabels([])
# axes[3,2].set_xlabel('lambda={:.2E}, $\mu$={:.2E}'.format(0.01953125,0.03125))
# axes[4,2].imshow(np.load('VAE_8_dim_no_sigmoid_z_sparse_one_lambda_0.0048828125_mu_0.015625.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# axes[4,2].set_yticklabels([])
# axes[4,2].set_xticklabels([])
# axes[4,2].set_xlabel('lambda={:.2E}, $\mu$={:.2E}'.format(0.0048828125,0.015625))


# axes[1,3].imshow(np.load('AE_8_dim_no_sigmoid_x_tik_one_lambda_2.5.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# axes[1,3].set_yticklabels([])
# axes[1,3].set_xticklabels([])
# axes[1,3].set_xlabel('lambda={:.2E}'.format(2.5))
# axes[2,3].imshow(np.load('AE_8_dim_no_sigmoid_x_tik_one_lambda_0.15625.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# axes[2,3].set_yticklabels([])
# axes[2,3].set_xticklabels([])
# axes[2,3].set_xlabel('lambda={:.2E}'.format(0.15625))
# axes[3,3].imshow(np.load('AE_8_dim_no_sigmoid_x_tik_one_lambda_0.01953125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# axes[3,3].set_yticklabels([])
# axes[3,3].set_xticklabels([])
# axes[3,3].set_xlabel('lambda={:.2E}'.format(0.01953125))
# axes[4,3].imshow(np.load('AE_8_dim_no_sigmoid_x_tik_one_lambda_0.0048828125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# axes[4,3].set_yticklabels([])
# axes[4,3].set_xticklabels([])
# axes[4,3].set_xlabel('lambda={:.2E}'.format(0.0048828125))


# plt.savefig('VAE_mnist_8_one.png')

# #%%
# gan_x_tik=np.load('AE_8_dim_no_sigmoid_x_tik_oneloss.npy')
# gan_x_soft=np.load('GAN_8_dim_no_sigmoid_x_soft_onelambda_mu_loss.npy')
# gan_z_sparse=np.load('GAN_8_dim_no_sigmoid_z_sparse_one_lambda_mu_loss.npy')
# gan_z_opt=np.load('GAN_8_dim_no_sigmoid_z_optimisation_one__loss.npy')


# fig, axes=plt.subplots(5,4, figsize=(40,50))

# plot=gan_z_opt
# axes[0,0].scatter(plot[:,0], plot[:,1])
# axes[0,0].hlines(28*28*0.1**2, np.min(plot[:,0]), np.max(plot[:,0]))
# axes[0,0].set_xscale('log')
# axes[0,0].set_xlabel('lambda')
# axes[0,0].set_ylabel('Data discrepancy')
# axes[0,0].set_ylim(0,30)

# plot=gan_x_soft
# im=axes[0,1].scatter(plot[:,0], plot[:,2], c=plot[:,1])
# #plt.colorbar(im,cax=axes[0,1])
# axes[0,1].hlines(28*28*0.1**2, np.min(plot[:,0]), np.max(plot[:,0]))
# axes[0,1].set_xscale('log')
# axes[0,1].set_xlabel('lambda')
# axes[0,1].set_ylabel('Data discrepancy')
# axes[0,1].set_ylim(0,30)

# plot=gan_z_sparse
# im=axes[0,2].scatter(plot[:,0], plot[:,2], c=plot[:,1])
# #plt.colorbar(im,cax=axes[0,2])
# axes[0,2].hlines(28*28*0.1**2, np.min(plot[:,0]), np.max(plot[:,0]))
# axes[0,2].set_xscale('log')
# axes[0,2].set_xlabel('lambda')
# axes[0,2].set_ylabel('Data discrepancy')
# axes[0,2].set_ylim(0,30)

# plot=gan_x_tik
# axes[0,3].scatter(plot[:,0], plot[:,1])
# axes[0,3].hlines(28*28*0.1**2, np.min(plot[:,0]), np.max(plot[:,0]))
# axes[0,3].set_xscale('log')
# axes[0,3].set_xlabel('lambda')
# axes[0,3].set_ylabel('Data discrepancy')
# axes[0,3].set_ylim(0,30)

# axes[1,0].imshow(np.load('GAN_8_dim_no_sigmoid_z_optimisation_one_lambda_2.5.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# axes[1,0].set_yticklabels([])
# axes[1,0].set_xticklabels([])
# axes[1,0].set_xlabel('lambda={:.2E}'.format(2.5))
# axes[2,0].imshow(np.load('GAN_8_dim_no_sigmoid_z_optimisation_one_lambda_0.15625.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# axes[2,0].set_yticklabels([])
# axes[2,0].set_xticklabels([])
# axes[2,0].set_xlabel('lambda={:.2E}'.format(0.15625))
# axes[3,0].imshow(np.load('GAN_8_dim_no_sigmoid_z_optimisation_one_lambda_0.01953125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# axes[3,0].set_yticklabels([])
# axes[3,0].set_xticklabels([])
# axes[3,0].set_xlabel('lambda={:.2E}'.format(0.01953125))
# axes[4,0].imshow(np.load('GAN_8_dim_no_sigmoid_z_optimisation_one_lambda_0.0048828125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# axes[4,0].set_yticklabels([])
# axes[4,0].set_xticklabels([])
# axes[4,0].set_xlabel('lambda={:.2E}'.format(0.0048828125))

# axes[1,1].imshow(np.load('GAN_8_dim_no_sigmoid_x_soft_onelambda_2.5_mu_2.5.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# axes[1,1].set_yticklabels([])
# axes[1,1].set_xticklabels([])
# axes[1,1].set_xlabel('lambda={:.2E}, $\mu$={:.2E}'.format(2.5,2.5))
# axes[2,1].imshow(np.load('GAN_8_dim_no_sigmoid_x_soft_onelambda_0.15625_mu_0.3125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# axes[2,1].set_yticklabels([])
# axes[2,1].set_xticklabels([])
# axes[2,1].set_xlabel('lambda={:.2E}, $\mu$={:.2E}'.format(0.15625,0.3125))
# axes[3,1].imshow(np.load('GAN_8_dim_no_sigmoid_x_soft_onelambda_0.01953125_mu_0.078125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# axes[3,1].set_yticklabels([])
# axes[3,1].set_xticklabels([])
# axes[3,1].set_xlabel('lambda={:.2E}, $\mu$={:.2E}'.format(0.01953125,0.078125))
# axes[4,1].imshow(np.load('GAN_8_dim_no_sigmoid_x_soft_onelambda_0.0048828125_mu_0.01953125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# axes[4,1].set_yticklabels([])
# axes[4,1].set_xticklabels([])
# axes[4,1].set_xlabel('lambda={:.2E}, $\mu$={:.2E}'.format(0.0048828125,0.01953125))

# axes[1,2].imshow(np.load('GAN_8_dim_no_sigmoid_z_sparse_one_lambda_2.5_mu_2.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# axes[1,2].set_yticklabels([])
# axes[1,2].set_xticklabels([])
# axes[1,2].set_xlabel('lambda={:.2E}, $\mu$={:.2E}'.format(2.5,2))
# axes[2,2].imshow(np.load('GAN_8_dim_no_sigmoid_z_sparse_one_lambda_0.15625_mu_0.25.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# axes[2,2].set_yticklabels([])
# axes[2,2].set_xticklabels([])
# axes[2,2].set_xlabel('lambda={:.2E}, $\mu$={:.2E}'.format(0.15625,0.25))
# axes[3,2].imshow(np.load('GAN_8_dim_no_sigmoid_z_sparse_one_lambda_0.01953125_mu_0.03125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# axes[3,2].set_yticklabels([])
# axes[3,2].set_xticklabels([])
# axes[3,2].set_xlabel('lambda={:.2E}, $\mu$={:.2E}'.format(0.01953125,0.03125))
# axes[4,2].imshow(np.load('GAN_8_dim_no_sigmoid_z_sparse_one_lambda_0.0048828125_mu_0.015625.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# axes[4,2].set_yticklabels([])
# axes[4,2].set_xticklabels([])
# axes[4,2].set_xlabel('lambda={:.2E}, $\mu$={:.2E}'.format(0.0048828125,0.015625))





# axes[1,3].imshow(np.load('AE_8_dim_no_sigmoid_x_tik_one_lambda_2.5.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# axes[1,3].set_yticklabels([])
# axes[1,3].set_xticklabels([])
# axes[1,3].set_xlabel('lambda={:.2E}'.format(2.5))
# axes[2,3].imshow(np.load('AE_8_dim_no_sigmoid_x_tik_one_lambda_0.15625.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# axes[2,3].set_yticklabels([])
# axes[2,3].set_xticklabels([])
# axes[2,3].set_xlabel('lambda={:.2E}'.format(0.15625))
# axes[3,3].imshow(np.load('AE_8_dim_no_sigmoid_x_tik_one_lambda_0.01953125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# axes[3,3].set_yticklabels([])
# axes[3,3].set_xticklabels([])
# axes[3,3].set_xlabel('lambda={:.2E}'.format(0.01953125))
# axes[4,3].imshow(np.load('AE_8_dim_no_sigmoid_x_tik_one_lambda_0.0048828125.npy'), cmap='gray', vmin=-.2, vmax=1.2)
# axes[4,3].set_yticklabels([])
# axes[4,3].set_xticklabels([])
# axes[4,3].set_xlabel('lambda={:.2E}'.format(0.0048828125))

# plt.savefig('GAN_mnist_8_one.png')

#%%
aim=np.load('aim_mnist_convolution_one.npy')
data=np.load('data_mnist_convolution_one.npy')

# Plot figure with subplots of different sizes
fig = plt.figure(1)
# set up subplot grid
gridspec.GridSpec(1,2)
plt.subplot2grid((1,2), (0,0))
im=plt.imshow(aim, cmap='gray', vmin=-0.2, vmax=1.2)
plt.yticks([])
plt.xticks([])
#plt.colorbar(im)
plt.xlabel('Ground truth', fontsize=10)

plt.subplot2grid((1,2), (0,1))

im=plt.imshow(data, cmap='gray', vmin=-0.2, vmax=1.2)
plt.yticks([])
plt.xticks([])

plt.xlabel('Observed Data', fontsize=10)

cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.7])
fig.colorbar(im,cax=cbar_ax)

fig_name = 'mnist_convolution_one_aim_data.png'
fig.savefig(fig_name, bbox_inches='tight' )