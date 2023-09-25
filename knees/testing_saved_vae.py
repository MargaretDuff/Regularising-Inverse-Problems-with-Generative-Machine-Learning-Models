# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np 
import generative_model as test_class
import glob
import os
import tensorflow as tf




test_files=glob.glob('./checkpoints_no_sigmoid/**/checkpoint', recursive=True)
save_file=np.zeros((len(test_files),2), dtype='object')
for i, file in enumerate(test_files):
    file_split=file.split('/')
    latent_dim=int(file_split[-2].split('_')[1])
    try:
        architecture_number=int(file_split[-2].split('_')[0][-1])
        latent_dim=int(file_split[-2].split('_')[1])
        test_images=np.load('../datasets/knee-fastMRI/knee_fastMRI_test.npy')

        if architecture_number==2:
            import knees2_vae_test as vae
        elif architecture_number==4:
            import knees4_vae_test as vae
        elif architecture_number==5:
            import knees5_vae_test as vae
        elif architecture_number==7:
            import knees7_vae_test as vae
        elif architecture_number==6:
            print('Error, architecture 6 gradients not currently working')
            break
        else:
            print('Error, architecture number is:  '+str(architecture_number))
            break
    except:
        try:
            architecture_number=int(file_split[-2].split('_')[1])
            latent_dim=int(file_split[-2].split('_')[2])
            test_images=np.load('../datasets/knee-fastMRI/knee_fastMRI_test_128.npy')
            if architecture_number==128:
                import knees_128_128_vae_test as vae
            else:
                print('Error, architecture number is:  '+str(architecture_number))
                break
        except:
            print('Error- '+file_split)
    checkpoints=glob.glob(file[:-len('checkpoint')]+'*.index')
    checkpoints.sort(key=os.path.getmtime)
    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    vae_test=vae.kneesVAE(latent_dim ,checkpoints[-1][:-len('.index')], sess)
    _,loss=vae_test.latentSpaceValues(test_images[:50], restore=False)
    save_file[i,0]=file
    save_file[i,1]=np.mean(loss)/np.mean(np.linalg.norm(test_images, ord=2, axis=(1,2))**2)
    sess.close()
    print(save_file)
    np.save('test_knees_vae_with_differing_kl.npy', save_file)




