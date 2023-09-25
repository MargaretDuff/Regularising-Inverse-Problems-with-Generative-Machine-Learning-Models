# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np 
import generative_testing_class as test_class
import mnist_vae_test as vae
import glob
import os

if __name__=='__main__':
#    train_images=np.load('mnist_train_images.npy')
    test_images=np.load('mnist_test_images.npy')
    print('Loaded images')

test_files=glob.glob('../mnist_vae/**/checkpoints*-29800.index', recursive=True)
save_file=np.zeros((len(test_files),2), dtype='object')
for i, file in enumerate(test_files):
    file_split=file.split('/')
    latent_dim=int(file_split[-1][len('checkpoints'):-len('-29800.index')])
    vae_test=vae.mnistVAE(latent_dim ,file[:-len('.index')])
    _,loss=vae_test.latentSpaceValues(test_images[:100], restore=False)
    save_file[i,0]=file
    save_file[i,1]=np.mean(loss)
print(save_file)
np.save('test_mnist_vae_with_differing_kl.npy', save_file)

