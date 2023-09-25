# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 16:39:49 2020

@author: marga
"""




import numpy as np
import tensorflow as tf
import odl


import generative_testing_class as test_class
  
if __name__=='__main__':
    train_images=np.load('mnist_train_images.npy')
    test_images=np.load('mnist_test_images.npy')

#%%

class mnistGAN(test_class.generative_model):

    def lrelu(self,x, alpha=0.3):
            return tf.maximum(x, tf.multiply(x, alpha))
    def discriminator(X_in, keep_prob=1.0):
            activation = self.lrelu
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
    def tf_generator(self,Z, keep_prob=1.0):
            with tf.variable_scope("generator",  reuse=tf.AUTO_REUSE ):
                x = tf.layers.dense(Z, 25, activation=self.lrelu)
                x = tf.layers.dense(x, units=49, activation=self.lrelu)
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
                
    def __init__(self, latent_dim,checkpoint, session ):
        super().__init__((28,28), latent_dim)
        self.session=session #set tensorflow session
       


        self.Y    = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='Y')
        self.Y_flat = tf.reshape(self.Y, shape=[-1, 28 * 28])

        self.z = tf.placeholder(tf.float32, [None, latent_dim])
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')

        self.reconstruction=self.tf_generator(self.z, self.keep_prob)
        self.unreshaped = tf.reshape(self.reconstruction, [-1, 28*28])

        self.img_loss = tf.reduce_sum(tf.squared_difference(self.unreshaped, self.Y_flat), 1) 
        self.grad=tf.gradients(self.img_loss, self.z)[0]         
        saver = tf.train.Saver()
        saver.restore(self.session, checkpoint) #restore from checkpoint


         #Define the generator as an odl operator
        generator_domain=tf.placeholder(dtype=tf.float32, shape=[1, self.n_latent], name='generator_domain')
        generator_result=self.tf_generator(generator_domain)[0,:,:]
        self.generator=odl.contrib.tensorflow.TensorflowOperator(generator_domain, generator_result, sess=self.session)


        #Note the relevant odl spaces for the latent and image spaces
        self.z_space=self.generator.domain
        self.x_space=self.generator.range

        #In the case of a tomographic inverse problem, a slightly different image space and therefore generator is r\equired
        self.tomo_space=odl.uniform_discr([-14., -14.], [ 14.,  14.], (28, 28), dtype='float32')
        generator_domain_tomo=tf.placeholder(dtype=tf.float32, shape=[1, self.n_latent], name='generator_domain_tomo')
        generator_result_tomo=self.tf_generator(generator_domain_tomo)[0,:,:]
        self.generator_tomo=odl.contrib.tensorflow.TensorflowOperator(generator_domain_tomo, generator_result_tomo, range=self.tomo_space,  sess=self.session)

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
