# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 10:04:55 2020

@author: magd21
"""

import matplotlib.pyplot as plt
import numpy as np
import sys, getopt
#import cv2
from PIL import Image


#%%
latent_dim=10
restore=False
save=True
iteration_start=0
iteration_end=30000
base='../datasets/2d_shapes/2d_shapes_train_images_'
data_set=base+'none.npy'
name='none'

try:
     opts, args = getopt.getopt(sys.argv[1:],"hs:n:r:b:e:d:",["save=","n_latent=", "restore=\
", "iteration_start=", "iteration_end=", "data_set="])
except getopt.GetoptError:
     print('2d_shapes_wgan.py -s <True/False> -n <n_latent> -r <"restore file"> -b <iteration_st\
art> -e <iteration_end> -d <none/circle/rect>')
     sys.exit(2)
for opt, arg in opts:
     if opt == '-h':
         print('2d_shapes_wgan.py -s <True/False> -n <n_latent> -r <"restore file"> -b <iteratio\
n_start> -e <iteration_end>')
         sys.exit()
     elif opt in ("-s", "--save"):
         if arg=='True':
             save = True
         else:
             save=False
     elif opt in ("-n", "--n_latent"):
        latent_dim =int( arg)
     elif opt in ("-r", "--restore"):
         if arg=='True':
             restore = True
         else:
             restore=str(arg)
             print(restore)
     elif opt in ("-b", "--iteration_start"):
         iteration_start=int(arg)
     elif opt in ("-e", "--iteration_end"):
         iteration_end=int(arg)
     elif opt in ("-d", "--data_set"):
          name=str(arg)
          data_set=base+str(arg)+'.npy'


train_images=np.load(data_set)

save_file='checkpoints_ns_'+str(latent_dim)+'_'+name+'_repeat/2d_shapes_wgan_'+str(latent_dim)+'dim'
#%%


#%%

def new_batch(image_set, batch_size=100):
                numbers=np.random.choice(range(np.shape(image_set)[0]), size=batch_size, replace=False, p=None)
                return image_set[numbers,:]



#%%
#
def plot(samples):
    fig = plt.figure(figsize=(16, 16))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i in range(16):
        ax = plt.subplot(gs[i])
        plt.imshow( samples[i,:,:], cmap='gray', vmin=0, vmax=1)
    return fig
        
 #%%       
import tensorflow as tf         
tf.reset_default_graph()
session = tf.InteractiveSession()

reshaped_dim = [-1, 7, 7, 1]

def lrelu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))


with tf.name_scope('placeholders'):
    x_true = tf.placeholder(tf.float32, [None, 56,56])
    z = tf.placeholder(tf.float32, [None, latent_dim])
    aim=tf.placeholder(tf.float32, [None, 56,56])
    keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')
def generator(z, keep_prob):
    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
#        x = tf.layers.dense(z, 25, activation=lrelu)
        x = tf.layers.dense(z, units=49, activation=lrelu)
        x = tf.reshape(x, reshaped_dim)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=56*56, activation=None)
        img = tf.reshape(x, shape=[-1, 56, 56])
        return img
def discriminator(X_in, keep_prob):
    activation = lrelu
    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
        X = tf.reshape(X_in, shape=[-1, 56, 56, 1])
        x = tf.layers.conv2d(X, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.contrib.layers.flatten(x)
        prob = tf.layers.dense(x, units=1)
        return prob
    


x_generated = generator(z, keep_prob)
reconstruction=x_generated
d_true = discriminator(x_true, keep_prob)
d_generated = discriminator(x_generated, keep_prob)


with tf.name_scope('regularizer'):
    epsilon = tf.random_uniform([50, 1,1], 0.0, 1.0)
    x_hat = epsilon *  x_true + (1 - epsilon) *  x_generated
    d_hat = discriminator(x_hat, keep_prob)

    gradients = tf.gradients(d_hat, x_hat)[0]
    ddx = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=1))
    d_regularizer = tf.reduce_mean((ddx - 1.0) ** 2)
with tf.name_scope('loss'):
    g_loss = tf.reduce_mean(d_generated)
    d_loss = (tf.reduce_mean(d_true) - tf.reduce_mean(d_generated) +
              10 * d_regularizer)
with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0, beta2=0.9)
    g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
    g_train = optimizer.minimize(g_loss, var_list=g_vars)
    d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
    d_train = optimizer.minimize(d_loss, var_list=d_vars)
print('Model set up (hopefully)')

#%%


if restore==False:
    tf.global_variables_initializer().run()
elif restore==True:
     try:
          import glob
          import os
          checkpoints=glob.glob(save_file+'*.index')
          checkpoints.sort(key=os.path.getmtime)
          restore_file=checkpoints[-1][:-6]
          print(restore_file)
          print(restore_file[len(save_file)+1:])
          iteration_start=int(restore_file[len(save_file)+1:])
          
          saver=tf.train.Saver()
          saver.restore(session, restore_file)
          print('restored at iteration   '+str(iteration_start))
     except:
          tf.global_variables_initializer().run()

else:
    saver=tf.train.Saver()
    saver.restore(session, restore)
    
if save:
    saver = tf.train.Saver()
    if restore==False:
         saver.save(session,save_file, global_step=0, write_meta_graph=True)

#%%
if save:
    plot_hold_g=np.zeros(iteration_end)
    plot_hold_d=np.zeros(iteration_end)
    for i in range(iteration_start, iteration_end):
        images = new_batch(train_images, batch_size=50)

        z_train = np.random.randn(50, latent_dim)
        plot_hold_g[i]=g_loss.eval(feed_dict={z: z_train, keep_prob:1.0})
        plot_hold_d[i]=d_loss.eval(feed_dict={x_true: images, z: z_train, keep_prob:1.0})

        session.run(g_train, feed_dict={z: z_train, keep_prob: 0.8})
        for j in range(5):
            session.run(d_train, feed_dict={x_true: images, z: z_train, keep_prob: 0.8})

        if i % 100 == 0:
             print('iter={}/{}'.format(i, iteration_end))
#        if __name__ == "__main__":
#            z_validate = np.random.randn(16, latent_dim)
#            generated = reconstruction.eval(feed_dict={z: z_validate, keep_prob:1.0}).squeeze()
#            fig=plot(generated)
#            plt.show()
             print('Gradient loss is ', g_loss.eval(feed_dict={z: z_train, keep_prob:1.0}))    
             print('Discriminator loss is ', d_loss.eval(feed_dict={x_true: images, z: z_train, keep_prob:1.0})) 
             saver.save(session, save_file, global_step=i,write_meta_graph=False)

plt.plot(range(iteration_end), plot_hold_g)
plt.title('Generator Loss')
plt.savefig(save_file+'generator_loss.png')
plt.plot(range(iteration_end), plot_hold_d)
plt.title('Discriminator Loss')
plt.savefig(save_file+'discriminator_loss.png')  





z_validate = np.random.randn(16, latent_dim)
generated = reconstruction.eval(feed_dict={z: z_validate, keep_prob:1.0}).squeeze()
print('generated')
for i in range(16):
    # matplotlib.image.imsave(save_name+'_generated_'+str(i)+'.png', generated[i,:,:])
     Image.fromarray((generated[i,:,:]*255).astype(np.uint8)).save(save_file+'_generated_'+str(i)+'.png')





