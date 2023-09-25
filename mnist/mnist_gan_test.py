# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 16:39:49 2020

@author: marga
"""




import numpy as np
import tensorflow as tf



import generative_testing_class as test_class
  
if __name__=='__main__':
    train_images=np.load('mnist_train_images.npy')
    test_images=np.load('mnist_test_images.npy')

#%%

class mnistGAN(test_class.generative_model):
    def __init__(self, latent_dim,checkpoint ):
        super().__init__((28,28), latent_dim)
        tf.reset_default_graph()
        x_true = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='X')
        batch_size=16
        self.z = tf.placeholder(tf.float32, [None, latent_dim])
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')
        def lrelu(x, alpha=0.3):
            return tf.maximum(x, tf.multiply(x, alpha))
        def discriminator(X_in, keep_prob):
            activation = lrelu
            with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
                X = tf.reshape(X_in, shape=[-1, 28, 28, 1])
                x = tf.layers.conv2d(X, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
                x = tf.nn.dropout(x, keep_prob)
                x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
                x = tf.nn.dropout(x, keep_prob)
                x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
                x = tf.nn.dropout(x, keep_prob)
                x = tf.contrib.layers.flatten(x)
                prob = tf.layers.dense(x, units=1)
                return prob
        def generator(Z, keep_prob):
            with tf.variable_scope("generator", reuse=None):
                x = tf.layers.dense(Z, 25, activation=lrelu)
                x = tf.layers.dense(x, units=49, activation=lrelu)
                x = tf.reshape(x, [-1, 7, 7, 1])
                x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
                x = tf.nn.dropout(x, keep_prob)
                x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
                x = tf.nn.dropout(x, keep_prob)
                x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
                
                x = tf.contrib.layers.flatten(x)
                x = tf.layers.dense(x, units=28*28, activation=None)
                img = tf.reshape(x, shape=[-1, 28, 28])
                return img
                
            
        self.reconstruction=generator(self.z, self.keep_prob)
        d_true=discriminator(x_true, self.keep_prob)
        d_generated=discriminator(self.reconstruction, self.keep_prob)
        with tf.name_scope('regularizer'):
            epsilon = tf.random_uniform([batch_size, 1,1], 0.0, 1.0)
            x_hat = epsilon *  x_true + (1 - epsilon) *  self.reconstruction
            d_hat = discriminator(x_hat, self.keep_prob)
        
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
        
        self.session=tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.session, checkpoint)
        print('Model set up (hopefully)')
        self.Y=tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='Y')
        self.img_loss=tf.square(tf.norm(self.reconstruction-self.Y, ord=2))
        self.grad=tf.gradients(self.img_loss, self.z)[0]
        
        
    def generate(self, z):
        if np.shape(np.shape(z))==(2,):
            if np.shape(z)[1:]==(self.n_latent,):
                return(self.session.run(self.reconstruction, feed_dict={self.z: z, self.keep_prob:1.0}))
        elif np.shape(z)==(self.n_latent,):
            return(self.session.run(self.reconstruction, feed_dict={self.z: np.expand_dims(z, axis=0), self.keep_prob:1.0}))
        else:
            print('z dimensions must be (?,latent_dim)')
            raise 
            
    
    def encode(self, x):
        #A single image!!
        if np.shape(np.shape(x))==(3,):
            if np.shape(x)==(1,28,28):
                pass
        elif np.shape(x)==(28,28):
            x= np.expand_dims(x, axis=0)
        else:
            print('Image dimensions must be (1,28,28)')
            raise
        
        z,loss=self.gradientDescentBacktracking(np.random.normal(0,1,(10,self.n_latent)),x, 200, initial_L=20,upper=5, lower=0.6)
        a=np.argmin(loss)
        return(z[a,:])
            
        
        

            
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
        
        return(self.session.run(self.img_loss, feed_dict={self.z: z, self.keep_prob:1.0, self.Y:y}))

      
        
        
    def gradients(self, z, y): #calculate gradient wrt z of ||G(z)-y||^2
        if np.shape(np.shape(y))==(3,):
            if np.shape(y)==(1,28,28):
                y=np.repeat(y, np.shape(z)[0],axis=0)
            elif np.shape(y)[1:]==(28,28):
                pass                
        elif np.shape(y)==(28,28):
            y= np.expand_dims(y, axis=0)
            y=np.repeat(y, np.shape(z)[0],axis=0)
        else:
            print('Image dimensions must be (?,28,28)')
            print(np.shape(y))
            raise
        if np.shape(np.shape(z))==(2,):
            if np.shape(z)[1:]==(self.n_latent,):
                pass
        elif np.shape(z)==(self.n_latent,):
            z= np.expand_dims(z, axis=0)
        else:
            print('z dimensions must be (?,latent_dim)')
            raise 
        return(self.session.run(self.grad, feed_dict={self.z: z, self.keep_prob:1.0, self.Y:y}))
        
        
 
#%%
if __name__=='__main__':
    test=mnistGAN(16, './GAN_checkpoints16/mnist_GAN_16-39800')
    # test.elephant()
    test.encode(train_images[600])
