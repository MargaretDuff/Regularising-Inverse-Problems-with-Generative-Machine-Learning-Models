# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 11:15:10 2020

@author: magd21
"""

import numpy as np
import h5py

import PIL
import PIL.Image as Image
import matplotlib.pyplot as plt

#%%

hf = h5py.File('file1002423.h5', 'r')

print('Keys:', list(hf.keys()))
print('Attrs:', dict(hf.attrs))


volume_kspace = hf['kspace'][()]
print(volume_kspace.dtype)
print(volume_kspace.shape)

slice_kspace = volume_kspace[20] # Choosing the 20-th slice of this volume

#%%

def show_coils(data, slice_nums, cmap=None):
    fig = plt.figure()
    for i, num in enumerate(slice_nums):
        plt.subplot(1, len(slice_nums), i + 1)
        plt.imshow(data[num], cmap=cmap)
        
        

show_coils(np.log(np.abs(volume_kspace) + 1e-9), [0, 5, 10])  # This shows coils 0, 5 and 10


#%%

reconstruction_esc = hf['reconstruction_esc'][()]
print(reconstruction_esc .dtype)
print(reconstruction_esc .shape)


def show_image(data, slice_nums, cmap=None):
    fig = plt.figure()
    for i, num in enumerate(slice_nums):
        plt.subplot(1, len(slice_nums), i + 1)
        plt.imshow(data[num], cmap=cmap)
        
        

show_image(reconstruction_esc, [20, 5, 10])  # This shows coils 0, 5 and 10


#%%
reconstruction_rss = hf['reconstruction_rss'][()]
print(reconstruction_rss.dtype)
print(reconstruction_rss.shape)


        
        

show_image(reconstruction_rss, [20, 5, 10])  # This shows coils 0, 5 and 10

#%%


for j in [10,15,20]:
    image=Image.fromarray((reconstruction_rss[j]), 'F')
    image=image.resize((80, 80), resample=Image.ANTIALIAS)
    image=np.array(image.getdata()).reshape([80,80])
    plt.imshow(image)


#%%
    
images=np.load('knee_fastMRI_test.npy')
np.shape(images)
plt.imshow(images[80,:,:], cmap='gray')