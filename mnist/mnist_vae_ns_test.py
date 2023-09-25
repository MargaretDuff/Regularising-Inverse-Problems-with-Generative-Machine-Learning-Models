# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 14:53:10 2020

@author: magd21
"""


import numpy as np

import tensorflow as tf



import generative_testing_class as test_class
if __name__=='__main__':
  train_images=np.load('mnist_train_images.npy')
  test_images=np.load('mnist_test_images.npy')


#%%

class mnistVAE(test_class.generative_model):
    def __init__(self, latent_dim,checkpoint ):
        super().__init__((28,28), latent_dim)
        tf.reset_default_graph()
        self.X_in = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='X')
        self.Y    = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='Y')
        self.Y_flat = tf.reshape(self.Y, shape=[-1, 28 * 28])
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')
        self.dec_in_channels = 1
        self.reshaped_dim = [-1, 7, 7, self.dec_in_channels]
        self.inputs_decoder = 49 * self.dec_in_channels / 2
        def lrelu(x, alpha=0.3):
            return tf.maximum(x, tf.multiply(x, alpha))
        def encoder(X_in, keep_prob):
            activation = lrelu
            with tf.variable_scope("encoder", reuse=None):
                X = tf.reshape(X_in, shape=[-1, 28, 28, 1])
                x = tf.layers.conv2d(X, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
                x = tf.nn.dropout(x, keep_prob)
                x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
                x = tf.nn.dropout(x, keep_prob)
                x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
                x = tf.nn.dropout(x, keep_prob)
                x = tf.contrib.layers.flatten(x)
                mn = tf.layers.dense(x, units=self.n_latent)
                sd       = 0.5 * tf.layers.dense(x, units=self.n_latent)            
                epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], self.n_latent])) 
                z  = mn + tf.multiply(epsilon, tf.exp(sd))
            return z, mn, sd
        def decoder(sampled_z, keep_prob):
            with tf.variable_scope("decoder", reuse=None):
                x = tf.layers.dense(sampled_z, 25, activation=lrelu)
                x = tf.layers.dense(x, units=49, activation=lrelu)
                x = tf.reshape(x, self.reshaped_dim)
                x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
                x = tf.nn.dropout(x, keep_prob)
                x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
                x = tf.nn.dropout(x, keep_prob)
                x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
                
                x = tf.contrib.layers.flatten(x)
                x = tf.layers.dense(x, units=28*28, activation=None)
                img = tf.reshape(x, shape=[-1, 28, 28])
                return img
                
        self.sampled, self.mn, self.sd = encoder(self.X_in, self.keep_prob)
        self.dec = decoder(self.sampled, self.keep_prob)
        self.session=tf.Session()
        self.unreshaped = tf.reshape(self.dec, [-1, 28*28])
        self.img_loss = tf.reduce_sum(tf.squared_difference(self.unreshaped, self.Y_flat), 1)
        self.latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * self.sd - tf.square(self.mn) - tf.exp(2.0 * self.sd), 1)
        self.loss = tf.reduce_mean(self.img_loss + self.latent_loss)
        
        saver = tf.train.Saver()
        saver.restore(self.session, checkpoint)
        
        self.grad=tf.gradients(self.img_loss, self.sampled)[0]
        
    def generate(self, z):
        if np.shape(np.shape(z))==(2,):
            if np.shape(z)[1:]==(self.n_latent,):
                return(self.session.run(self.dec, feed_dict={self.sampled: z, self.keep_prob:1.0}))
        elif np.shape(z)==(self.n_latent,):
            return(self.session.run(self.dec, feed_dict={self.sampled: np.expand_dims(z, axis=0), self.keep_prob:1.0}))
        else:
            print('z dimensions must be (?,latent_dim)')
            raise 
            
    
    def encode(self, x):
        if np.shape(np.shape(x))==(3,):
            if np.shape(x)[1:]==(28,28):
                pass
        elif np.shape(x)==(28,28):
            x=np.expand_dims(x, axis=0)
        else:
            print('Image dimensions must be (?,28,28)')
            raise 
        z0=self.session.run(self.sampled, feed_dict={self.X_in: x, self.keep_prob:1.0})
        z,_=self.gradientDescentBacktracking(z0,x, 20, initial_L=20,upper=5, lower=0.6)
        return(z)
            
        
        



        

            
    def observationLoss(self, z, y): #calculate ||G(z)-y||^2
        if np.shape(np.shape(y))==(3,):
            if np.shape(y)==(1,28,28):
                pass
        elif np.shape(y)==(28,28):
            y= np.expand_dims(y, axis=0)
        else:
            print('Image dimensions must be (1,28,28)')
            raise
        if np.shape(np.shape(z))==(2,):
            if np.shape(z)[1:]==(self.n_latent,):
                pass
        elif np.shape(z)==(self.n_latent,):
            z= np.expand_dims(z, axis=0)
        else:
            print('z dimensions must be (?,latent_dim)')
            raise 
        return(self.session.run(self.img_loss, feed_dict={self.sampled: z,self.Y:y, self.keep_prob:1.0}))
      
        
        
    def gradients(self, z, y): #calculate gradient wrt z of ||G(z)-y||^2
        if np.shape(np.shape(y))==(3,):
            if np.shape(y)==(1,28,28):
                pass
        elif np.shape(y)==(28,28):
            y= np.expand_dims(y, axis=0)
        else:
            print('Image dimensions must be (1,28,28)')
            raise
        if np.shape(np.shape(z))==(2,):
            if np.shape(z)[1:]==(self.n_latent,):
                pass
        elif np.shape(z)==(self.n_latent,):
            z= np.expand_dims(z, axis=0)
        else:
            print('z dimensions must be (?,latent_dim)')
            raise 
        return(self.session.run(self.grad, feed_dict={self.sampled: z,self.Y:y, self.keep_prob:1.0}))
        
        
 
#%%
if __name__=='__main__':
    test=mnistVAE(8, './VAE_Testing/checkpoints8/mnist_VAE-29800')
    # test.elephant()
    # test.encode(train_images[600])
    test.histInOutLoss( [test_images], ['Test_data'], bins=None)
