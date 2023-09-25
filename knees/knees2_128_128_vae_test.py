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


#%%

class kneesVAE(test_class.generative_model):
    def lrelu(self,x, alpha=0.3):
        return tf.maximum(x, tf.multiply(x, alpha))
    def tf_encoder(self, X_in, keep_prob=1.0):
        activation = self.lrelu

        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
          X = tf.reshape(X_in, shape=[-1, 128, 128, 1])
          x = tf.layers.conv2d(X, filters=8, kernel_size=3, strides=1, padding='same', activation=activation)
          x = tf.nn.dropout(x, keep_prob)
          x = tf.layers.conv2d(x, filters=16, kernel_size=3, strides=1, padding='same', activation=activation)
          x = tf.nn.dropout(x, keep_prob)
          x = tf.layers.conv2d(x, filters=32, kernel_size=3, strides=1, padding='same', activation=activation)
          x = tf.nn.dropout(x, keep_prob)
          
          x = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=2, padding='same', activation=activation)
          x = tf.nn.dropout(x, keep_prob)
          x = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=1, padding='same', activation=activation)
          x = tf.nn.dropout(x, keep_prob)
          x = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=1, padding='same', activation=activation)
          x = tf.nn.dropout(x, keep_prob)
          
          x = tf.layers.conv2d(x, filters=128, kernel_size=3, strides=2, padding='same', activation=activation)
          x = tf.nn.dropout(x, keep_prob)
          x = tf.layers.conv2d(x, filters=128, kernel_size=3, strides=1, padding='same', activation=activation)
          x = tf.nn.dropout(x, keep_prob)
          x = tf.layers.conv2d(x, filters=128, kernel_size=3, strides=1, padding='same', activation=activation)
          x = tf.nn.dropout(x, keep_prob)
          
          
          x = tf.layers.conv2d(x, filters=256, kernel_size=3, strides=2, padding='same', activation=activation)
          x = tf.nn.dropout(x, keep_prob)
          x = tf.layers.conv2d(x, filters=256, kernel_size=3, strides=1, padding='same', activation=activation)
          x = tf.nn.dropout(x, keep_prob)
          x = tf.layers.conv2d(x, filters=256, kernel_size=3, strides=1, padding='same', activation=activation)
          x = tf.nn.dropout(x, keep_prob)
          
          x = tf.layers.conv2d(x, filters=512, kernel_size=3, strides=2, padding='same', activation=activation)
          x = tf.nn.dropout(x, keep_prob)
          x = tf.layers.conv2d(x, filters=512, kernel_size=3, strides=1, padding='same', activation=activation)
          x = tf.nn.dropout(x, keep_prob)
          x = tf.layers.conv2d(x, filters=512, kernel_size=3, strides=1, padding='same', activation=activation)
          x = tf.nn.dropout(x, keep_prob)
          
          x = tf.layers.conv2d(x, filters=128, kernel_size=3, strides=1, padding='same', activation=activation)
          x = tf.nn.dropout(x, keep_prob)
          
          x = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=1, padding='same', activation=activation)
          x = tf.nn.dropout(x, keep_prob)
          x = tf.layers.conv2d(x, filters=32, kernel_size=3, strides=1, padding='same', activation=activation)
          x = tf.nn.dropout(x, keep_prob)
          x = tf.layers.conv2d(x, filters=16, kernel_size=3, strides=1, padding='same', activation=activation)
          x = tf.nn.dropout(x, keep_prob)

          x = tf.contrib.layers.flatten(x)
          mn = tf.layers.dense(x, units=self.n_latent)
          sd       = 0.5 * tf.layers.dense(x, units=self.n_latent)
          epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], self.n_latent]))
          z  = mn + tf.multiply(epsilon, tf.exp(sd))

            
          return z, mn, sd
    
    def tf_generator(self, sampled_z, keep_prob=1.0):
        activation=tf.nn.relu
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            x = tf.layers.dense(sampled_z, units=1024, activation=activation)
            print(x)
            x = tf.reshape(x, [-1,8,8,16])
            print(x)


            x = tf.layers.conv2d_transpose(x, filters=32, kernel_size=3, strides=1, padding='same', activation=activation)
            x = tf.nn.dropout(x, keep_prob)
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=3, strides=1, padding='same', activation=activation)
            x = tf.nn.dropout(x, keep_prob)
            x = tf.layers.conv2d_transpose(x, filters=128, kernel_size=3, strides=1, padding='same', activation=activation)
            x = tf.nn.dropout(x, keep_prob)
            x = tf.layers.conv2d_transpose(x, filters=356, kernel_size=3, strides=1, padding='same', activation=activation)
            x = tf.nn.dropout(x, keep_prob)
            

            x = tf.layers.conv2d_transpose(x, filters=512, kernel_size=3, strides=1, padding='same', activation=activation)
            x = tf.nn.dropout(x, keep_prob)
            x = tf.layers.conv2d_transpose(x, filters=512, kernel_size=3, strides=1, padding='same', activation=activation)
            x = tf.nn.dropout(x, keep_prob)
            x = tf.layers.conv2d_transpose(x, filters=512, kernel_size=3, strides=1, padding='same', activation=activation)
            x = tf.nn.dropout(x, keep_prob)
            x = tf.layers.conv2d_transpose(x, filters=512, kernel_size=3, strides=1, padding='same', activation=activation)
            x = tf.nn.dropout(x, keep_prob)




            x = tf.layers.conv2d_transpose(x, filters=256, kernel_size=3, strides=2, padding='same', activation=activation)
            x = tf.nn.dropout(x, keep_prob)
            x = tf.layers.conv2d_transpose(x, filters=256, kernel_size=3, strides=1, padding='same', activation=activation)
            x = tf.nn.dropout(x, keep_prob)
            x = tf.layers.conv2d_transpose(x, filters=256, kernel_size=3, strides=1, padding='same', activation=activation)
            x = tf.nn.dropout(x, keep_prob)
            x = tf.layers.conv2d_transpose(x, filters=256, kernel_size=3, strides=1, padding='same', activation=activation)
            x = tf.nn.dropout(x, keep_prob)
            
            
            x = tf.layers.conv2d_transpose(x, filters=128, kernel_size=3, strides=2, padding='same', activation=activation)
            x = tf.nn.dropout(x, keep_prob)
            x = tf.layers.conv2d_transpose(x, filters=128, kernel_size=3, strides=1, padding='same', activation=activation)
            x = tf.nn.dropout(x, keep_prob)
            x = tf.layers.conv2d_transpose(x, filters=128, kernel_size=3, strides=1, padding='same', activation=activation)
            x = tf.nn.dropout(x, keep_prob)
            x = tf.layers.conv2d_transpose(x, filters=128, kernel_size=3, strides=1, padding='same', activation=activation)
            x = tf.nn.dropout(x, keep_prob)
            
            
            
            
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=3, strides=2, padding='same', activation=activation)
            x = tf.nn.dropout(x, keep_prob)
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=3, strides=1, padding='same', activation=activation)
            x = tf.nn.dropout(x, keep_prob)
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=3, strides=1, padding='same', activation=activation)
            x = tf.nn.dropout(x, keep_prob)
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=3, strides=1, padding='same', activation=activation)
            x = tf.nn.dropout(x, keep_prob)
            
            x = tf.layers.conv2d_transpose(x, filters=32, kernel_size=3, strides=2, padding='same', activation=activation)
            x = tf.nn.dropout(x, keep_prob)
            x = tf.layers.conv2d_transpose(x, filters=16, kernel_size=3, strides=1, padding='same', activation=activation)
            x = tf.nn.dropout(x, keep_prob)
            x = tf.layers.conv2d_transpose(x, filters=8, kernel_size=3, strides=1, padding='same', activation=activation)
            x = tf.nn.dropout(x, keep_prob)
            x = tf.layers.conv2d_transpose(x, filters=4, kernel_size=3, strides=1, padding='same', activation=activation)
            x = tf.nn.dropout(x, keep_prob)
            
            x = tf.layers.conv2d_transpose(x, filters=1, kernel_size=3, strides=1, padding='same', activation=tf.nn.tanh)
            x = tf.nn.dropout(x, keep_prob)
            x=tf.contrib.layers.flatten(x)
            
            img = tf.reshape(x, shape=[-1, 128, 128])
            return img








    def __init__(self, latent_dim,checkpoint, session ):
        super().__init__((128,128), latent_dim)
      
        self.session=session
        self.X_in = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128], name='X')
        self.Y    = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128], name='Y')
        self.Y_flat = tf.reshape(self.Y, shape=[-1, 128 * 128])
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')
       
       
   

                
        self.z, self.mn, self.sd = self.tf_encoder(self.X_in, self.keep_prob)
        self.reconstruction = self.tf_generator(self.z, self.keep_prob)
        self.unreshaped = tf.reshape(self.reconstruction, [-1, 128*128])
        self.img_loss = tf.reduce_sum(tf.squared_difference(self.unreshaped, self.Y_flat), 1)
            
        saver = tf.train.Saver()
        saver.restore(self.session, checkpoint)
        
        self.grad=tf.gradients(self.img_loss, self.z)[0]
        
        banana=tf.placeholder(dtype=tf.float32, shape=[1, self.n_latent], name='banana')
        rabbit=self.tf_generator(banana)[0,:,:]
        self.generator=odl.contrib.tensorflow.TensorflowOperator(banana, rabbit, sess=self.session)
        
        orange=tf.placeholder(dtype=tf.float32, shape=[ 128, 128], name='orange')
        _, fox,_=self.tf_encoder(orange)
        self.encoder=odl.contrib.tensorflow.TensorflowOperator(orange, fox,  sess=self.session)
        
        
        self.z_space=self.generator.domain
        self.x_space=self.generator.range
        
        self.tomo_space=odl.uniform_discr([-64., -64.], [ 64.,  64.], (128, 128), dtype='float32')
        banana_tomo=tf.placeholder(dtype=tf.float32, shape=[1, self.n_latent], name='banana')
        rabbit_tomo=self.tf_generator(banana_tomo)[0,:,:]
        self.generator_tomo=odl.contrib.tensorflow.TensorflowOperator(banana_tomo, rabbit_tomo, range=self.tomo_space,  sess=self.session)


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
        if np.shape(np.shape(x))==(3,):
            if np.shape(x)[1:]==(128,128):
                pass
        elif np.shape(x)==(128,128):
            x=np.expand_dims(x, axis=0)
        else:
            print('Image dimensions must be (?,80,80)')
            raise 
        z0=self.session.run(self.z, feed_dict={self.X_in: x, self.keep_prob:1.0})
        z,_=self.gradientDescentBacktracking(z0,x, 50, initial_L=20,upper=5, lower=0.6)
        return(z)
            
        
        



        

            
    def observationLoss(self, z, y): #calculate ||G(z)-y||^2
        if np.shape(np.shape(y))==(3,):
            if np.shape(y)==(1,128,128):
                pass
        elif np.shape(y)==(128,128):
            y= np.expand_dims(y, axis=0)
        else:
            print('Image dimensions must be (1,128,128)')
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
      
        
        
    def gradients(self, z, y): #calculate gradient wrt z of ||G(z)-y||^2
        if np.shape(np.shape(y))==(3,):
            if np.shape(y)==(1,128,128):
                pass
        elif np.shape(y)==(128,128):
            y= np.expand_dims(y, axis=0)
        else:
            print('Image dimensions must be (1,128,128)')
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
        
        
 
#%%
if __name__=='__main__':
    test=mnistVAE(8, './VAE_Testing/checkpoints8/mnist_VAE-29800')
    # test.elephant()
    # test.encode(train_images[600])
    test.histInOutLoss( [test_images], ['Test_data'], bins=None)
