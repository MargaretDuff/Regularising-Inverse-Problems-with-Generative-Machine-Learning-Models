# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np 
import generative_model as test_class
import vae_2d_shapes_class  as vae
import glob
import os
import tensorflow as tf

if __name__=='__main__':
#    train_images=np.load('mnist_train_images.npy')
    test_images=np.load('../../datasets/2d_shapes/2d_shapes_test_images_none.npy')
    print('Loaded images')

test_files=glob.glob('../../2d_shapes_vae/checkpoints_no_sigmoid/**/2d_shapes_VAE-29800.index', recursive=True)
save_file=np.zeros((len(test_files),2), dtype='object')
for i, file in enumerate(test_files):
    file_split=file.split('/')
    latent_dim=int(file_split[-2].split('_')[2])
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    vae_test=vae.shapes_VAE(latent_dim ,file[:-len('.index')], sess)
    _,loss=vae_test.latentSpaceValues(test_images[:100], restore=False)
    sess.close()
    save_file[i,0]=file
    save_file[i,1]=np.mean(loss)
print(save_file)
np.save('test_shapes_vae_with_differing_kl.npy', save_file)

