# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 14:53:10 2020

@author: magd21
"""


import numpy as np
import tensorflow as tf
import odl 
import odl.contrib.tensorflow

import generative_model as test_class
#from gd_methods import gradientDescentBacktracking

#%%

class mnistVAE(test_class.generative_model):
    
    
    def lrelu(self,x, alpha=0.3): # Define leaky ReLU activation function to be used in the encoder 
        return tf.maximum(x, tf.multiply(x, alpha))
    
    
    def tf_encoder(self,X_in, keep_prob=1.0): #the encoder architecture 
        activation = self.lrelu
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            X = tf.reshape(X_in, shape=[-1, 28, 28, 1])
            x = tf.layers.conv2d(X, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
            x = tf.nn.dropout(x, keep_prob)
            x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
            x = tf.nn.dropout(x, keep_prob)
            x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
            x = tf.nn.dropout(x, keep_prob)
            
            x = tf.contrib.layers.flatten(x)
            mn = tf.layers.dense(x, units=self.n_latent) #encode a mean in the latent space
            sd       = 0.5 * tf.layers.dense(x, units=self.n_latent)  #encode a variance in the latent space      
            epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], self.n_latent])) 
            z  = mn + tf.multiply(epsilon, tf.exp(sd)) #the reparmaeterisation trick
        return z, mn, sd
    
    
    def tf_generator(self,sampled_z, keep_prob=1.0): # the decoder architecture 
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            x = tf.layers.dense(sampled_z, 25, activation=self.lrelu)
            x = tf.layers.dense(x, units=49, activation=self.lrelu)
            x = tf.reshape(x, [-1, 7, 7, 1])
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
            x = tf.nn.dropout(x, keep_prob)
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
            x = tf.nn.dropout(x, keep_prob)
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
            
            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, units=28*28, activation=None) #No activation in the last layer - we don't want to put any limits or assumptions on the range of the output images 
            img = tf.reshape(x, shape=[-1, 28, 28])
            return img
        
        
    def __init__(self, latent_dim,checkpoint, session ):
        super().__init__((28,28), latent_dim)
        
        self.session=session #set tensorflow session
        
        # Initialise tensorflow placeholders
        self.X_in = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='X')
        self.Y    = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='Y')
        self.Y_flat = tf.reshape(self.Y, shape=[-1, 28 * 28])
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob') # for dropout layers 
        

                
        self.z, self.mn, self.sd = self.tf_encoder(self.X_in, self.keep_prob) #define the encoder
        self.reconstruction = self.tf_generator(self.z, self.keep_prob) # link the encoder and the decoder 
        
        #Compute the losses for the VAE
        self.unreshaped = tf.reshape(self.reconstruction, [-1, 28*28])
        self.img_loss = tf.reduce_sum(tf.squared_difference(self.unreshaped, self.Y_flat), 1) #l2 norm squared loss between image and reconstruction 
        self.latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * self.sd - tf.square(self.mn) - tf.exp(2.0 * self.sd), 1) #KL loss between encoding distribution and a N(0,1) prior 
        self.loss = tf.reduce_mean(self.img_loss + self.latent_loss) # total loss 
        
        saver = tf.train.Saver()
        saver.restore(self.session, checkpoint) #restore from checkpoint 
        
        self.grad=tf.gradients(self.img_loss, self.z)[0] # tensorflow gradients 
        
        #Define the generator as an odl operator
        generator_domain=tf.placeholder(dtype=tf.float32, shape=[1, self.n_latent], name='generator_domain')
        generator_result=self.tf_generator(generator_domain)[0,:,:]
        self.generator=odl.contrib.tensorflow.TensorflowOperator(generator_domain, generator_result, sess=self.session)
        
        #Define the encoder as an odl operator
        encoder_domain=tf.placeholder(dtype=tf.float32, shape=[ 28, 28], name='encoder_domain')
        _, encoder_result,_=self.tf_encoder(encoder_domain)
        self.encoder=odl.contrib.tensorflow.TensorflowOperator(encoder_domain, encoder_result,  sess=self.session)
        
        #Note the relevant odl spaces for the latent and image spaces
        self.z_space=self.generator.domain
        self.x_space=self.generator.range
        
        #In the case of a tomographic inverse problem, a slightly different image space and therefore generator is required
        self.tomo_space=odl.uniform_discr([-14., -14.], [ 14.,  14.], (28, 28), dtype='float32')
        generator_domain_tomo=tf.placeholder(dtype=tf.float32, shape=[1, self.n_latent], name='generator_domain_tomo')
        generator_result_tomo=self.tf_generator(generator_domain_tomo)[0,:,:]
        self.generator_tomo=odl.contrib.tensorflow.TensorflowOperator(generator_domain_tomo, generator_result_tomo, range=self.tomo_space,  sess=self.session)


    def generate(self, z): # A function that takes a numpy array of latent vectors, outputting the resulting generated images as a numpy array 
        if np.shape(np.shape(z))==(2,):
            if np.shape(z)[1:]==(self.n_latent,):
                return(self.session.run(self.reconstruction, feed_dict={self.z: z, self.keep_prob:1.0}))
        elif np.shape(z)==(8,):
            return(self.session.run(self.reconstruction, feed_dict={self.z: np.expand_dims(z, axis=0), self.keep_prob:1.0}))
        else:
            print('z dimensions must be (?,latent_dim)')
            raise 
            
    
    def encode(self, x): # A function that takes a numpy array of images, outputting a possible approximation of the resulting latent vectors that map to these images
       
        if np.shape(np.shape(x))==(3,):
            if np.shape(x)[1:]==(28,28):
                pass
        elif np.shape(x)==(28,28):
            x=np.expand_dims(x, axis=0)
        else:
            print('Image dimensions must be (?,28,28)')
            raise 
        z0=self.session.run(self.z, feed_dict={self.X_in: x, self.keep_prob:1.0})
        z,_=gradientDescentBacktracking(z0,x, 20, initial_L=20,upper=5, lower=0.6) #the encoder is not designed to be an inverse of the generator. We take the encoded values as an itialisaiton of a gradient based method for minimising ||G(z)-y||_2^2
        return(z)
            
        
        



        

            
    def observationLoss(self, z, y): #calculate ||G(z)-y||^2 for numpy arrays y and z 
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
        return(self.session.run(self.img_loss, feed_dict={self.z: z,self.Y:y, self.keep_prob:1.0}))
      
        
        
    def gradients(self, z, y): #calculate gradient wrt z of ||G(z)-y||^2 for numpy arrays y and z. Outputs a numpy array.
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
        return(self.session.run(self.grad, feed_dict={self.z: z,self.Y:y, self.keep_prob:1.0}))
        
        
 

