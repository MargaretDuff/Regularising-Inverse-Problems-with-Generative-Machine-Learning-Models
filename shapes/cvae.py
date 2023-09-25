

# ##### Copyright 2018 The TensorFlow Authors.

# In[1]:


#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# In[3]:
import tensorflow as tf
import numpy as np

import sys, getopt
from PIL import Image
import os



def new_batch(image_set, batch_size=100):
                numbers=np.random.choice(range(np.shape(image_set)[0]), size=batch_size, replace=False, p=None)
                return image_set[numbers,:]

# %%
batch_size = 32


X_in = tf.placeholder(dtype=tf.float32, shape=[None, 56, 56], name='X')
Y    = tf.placeholder(dtype=tf.float32, shape=[None, 56, 56], name='Y')
Y_flat = tf.reshape(Y, shape=[-1, 56 * 56])
keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')

n_latent=10
dec_in_channels = 1
restore=True
save=True
iteration_start=0
iteration_end=30000
base='../datasets/2d_shapes/2d_shapes_train_images_'
data_set=base+'none.npy'
name='none'
kl_factor=1

try:
     opts, args = getopt.getopt(sys.argv[1:],"hs:n:r:b:e:d:k:",["save=","n_latent=", "restore=", "iteration_start=", "iteration_", "kl_factor="])
except getopt.GetoptError:
     print('cvae.py -s <True/False> -n <n_latent> -r <"restore file"> -b <iteration_start> -e <iteration_end> -d <none/circle/rect> -k <kl_factor>')
     sys.exit(2)
for opt, arg in opts:
     if opt == '-h':
         print('cvae.py  -s <True/False> -n <n_latent> -r <"restore file"> -b <iteration_start> -e <iteration_end> -d <none/circle/rect> -k <kl_factor>')
         sys.exit()
     elif opt in ("-s", "--save"):
         if arg=='True':
             save = True
         else:
             save=False 
     elif opt in ("-n", "--n_latent"):
        n_latent =int( arg)
     elif opt in ("-r", "--restore"):
          if arg=='True':
              restore = True
          else:
              restore=str(arg)
              print(restore)
     elif opt in ("-k","--kl_factor"):
          kl_factor=float(arg)

     elif opt in ("-b", "--iteration_start"):
         iteration_start=int(arg)
     elif opt in ("-e", "--iteration_end"):
         iteration_end=int(arg)
     elif opt in ("-d", "--data_set"):
          name=str(arg)
          data_set=base+str(arg)+'.npy'
        
        
train_images=np.load(data_set)

tf.reset_default_graph()
sess = tf.InteractiveSession()


X_in = tf.placeholder(dtype=tf.float32, shape=[None, 56, 56], name='X')
Y    = tf.placeholder(dtype=tf.float32, shape=[None, 56, 56], name='Y')
Y_flat = tf.reshape(Y, shape=[-1, 56 * 56])
keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')


reshaped_dim = [-1, 7, 7, dec_in_channels]


def lrelu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))

#%%

def encoder(X_in, keep_prob=1.0):
    activation = lrelu
    with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
        X = tf.reshape(X_in, shape=[-1, 56, 56, 1])
        x = tf.layers.conv2d(X, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        #x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        #x = tf.nn.dropout(x, keep_prob)
        x = tf.contrib.layers.flatten(x)
        mn = tf.layers.dense(x, units=n_latent)
        sd       = 0.5 * tf.layers.dense(x, units=n_latent)            
        epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_latent])) 
        z  = mn + tf.multiply(epsilon, tf.exp(sd))
        
        return z, mn, sd
    
#%%
def decoder(sampled_z, keep_prob=1.0):
     activation=lrelu
     with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
          x = tf.layers.dense(sampled_z, units=49, activation=lrelu)
          x = tf.reshape(x, reshaped_dim)
          #x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', activation=lrelu)
          #x = tf.nn.dropout(x, keep_prob)
          x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', activation=lrelu)
          x = tf.nn.dropout(x, keep_prob)
          x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', activation=lrelu)
          x = tf.nn.dropout(x, keep_prob)
          x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=lrelu)
          x = tf.nn.dropout(x, keep_prob)
          x=tf.contrib.layers.flatten(x)
          x=tf.layers.dense(x, units=56*56, activation=None)
          img = tf.reshape(x, shape=[-1, 56, 56])
          return img
    
#%%
sampled, mn, sd = encoder(X_in, keep_prob)
dec = decoder(sampled, keep_prob)

#%%
unreshaped = tf.reshape(dec, [-1, 56*56])
img_loss = tf.reduce_sum(tf.squared_difference(unreshaped, Y_flat), 1)
latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)
loss = tf.reduce_mean(img_loss + kl_factor*latent_loss)
optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)

#%%
sess = tf.Session()


#%%
saver = tf.train.Saver()
save_file='./checkpoints_no_sigmoid/checkpoints_'+name+'_'+str(n_latent)+'_kl_factor_'+str(kl_factor)+'/2d_shapes_VAE'



if restore==False:
    sess.run(tf.global_variables_initializer())
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
          saver.restore(sess, restore_file)
          print('restored at iteration   '+str(iteration_start))
     except:
          sess.run(tf.global_variables_initializer())

else:
     saver=tf.train.Saver()
     saver.restore(session, restore)


if save:
     saver.save(sess, save_file, global_step=iteration_start, write_meta_graph=True)

def tile_images(image_stack):
    """Given a stacked tensor of images, reshapes them into a horizontal tiling for
    display."""
    assert len(image_stack.shape) == 3
    image_list = [image_stack[i, :, :] for i in range(image_stack.shape[0])]
    tiled_images = np.concatenate(image_list, axis=1)
    return tiled_images


def generate_images( epoch):
    """Feeds random seeds into the generator and tiles and saves the output to a PNG
    file."""

    test_image_stack = dec.eval(session=sess, feed_dict={sampled: np.random.rand(10, n_latent),keep_prob:1.0 })
    test_image_stack = (test_image_stack -np.min(test_image_stack))
    test_image_stack=test_image_stack*255/np.max(test_image_stack)
    test_image_stack = np.squeeze(np.round(test_image_stack).astype(np.uint8))
    tiled_output = tile_images(test_image_stack)
    tiled_output = Image.fromarray(tiled_output, mode='L')  # L specifies greyscale
    outfile = os.path.join('./checkpoints_no_sigmoid/checkpoints_'+name+'_'+str(n_latent)+'_kl_factor_'+str(kl_factor), 'epoch_{}.png'.format(epoch))
    tiled_output.save(outfile)

#%%
if save:
     for i in range(iteration_start, iteration_end):
        batch = [np.reshape(b, [56, 56]) for b in new_batch(train_images, batch_size)]
        sess.run(optimizer, feed_dict = {X_in: batch, Y: batch, keep_prob: 0.8})
            
        if not i % 200:
            saver.save(sess, save_file, global_step=i,write_meta_graph=False)
                            
            ls, d, i_ls, d_ls, mu, sigm = sess.run([loss, dec, img_loss, latent_loss, mn, sd], feed_dict = {X_in: batch, Y: batch, keep_prob: 1.0})
            generate_images(i)
            print(i, ls, np.mean(i_ls), np.mean(d_ls))
#        
#%%
        
#randoms = [np.random.normal(0, 1, n_latent) for _ in range(10)]
#imgs = sess.run(dec, feed_dict = {sampled: randoms, keep_prob: 1.0})
#imgs = [np.reshape(imgs[i], [56, 56]) for i in range(len(imgs))]

#for i, img in  enumerate(imgs):
#    cv2.imwrite('./checkpoints_no_sigmoid/checkpoints_'+name+'_'+str(n_latent)+'/vae_generated_'+str(n_latent)+'dim_'+str(i)+'.png', img*256)

