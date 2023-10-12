##https://towardsdatascience.com/teaching-a-variational-autoencoder-vae-to-draw-mnist-characters-978675c95776



# In[3]:
import tensorflow as tf
import numpy as np
import sys, getopt
#import cv2
from PIL import Image
import os

kl_factor=1
n_latent=8
dec_in_channels = 1
restore=False
save=True
iteration_start=0
iteration_end=30000

try:
    opts, args = getopt.getopt(sys.argv[1:],"hs:n:r:b:e:k:",["save=","n_latent=", "restore=", "iteration_start=", "iteration_end=", "kl_regularisation="])
except getopt.GetoptError:
    print('cvae_no_sigmoid.py -s <True/False> -n <n_latent> -r <"restore file"> -b <iteration_start> -e <iteration_end> -k <kl_factor>')
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print('cvae_no_sigmoid.py -s <True/False> -n <n_latent> -r <"restore file"> -b <iteration_start> -e <iteration_end> -k <kl_factor>')
        sys.exit()
    elif opt in ("-s", "--save"):
        if arg=='True':
            save = True
        else:
            save=False 
    elif opt in ("-n", "--n_latent"):
       n_latent =int( arg)
    elif opt in ("-k", "--kl_factor"):
        kl_factor=float(arg)
    elif opt in ("-b", "--iteration_start"):
         iteration_start=int(arg)
    elif opt in ("-e", "--iteration_end"):
         iteration_end=int(arg)
    elif opt in ("-r", "--restore"):
          if arg=='True':
              restore = True
          else:
              restore=str(arg)
              print(restore)

    



train_images=np.load('mnist_train_images.npy')




def new_batch(image_set, batch_size=100):
                numbers=np.random.choice(range(np.shape(image_set)[0]), size=batch_size, replace=False, p=None)
                return image_set[numbers,:]

# %%
tf.reset_default_graph()

batch_size = 64


X_in = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='X')
Y    = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='Y')
Y_flat = tf.reshape(Y, shape=[-1, 28 * 28])
keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')

dec_in_channels = 1

reshaped_dim = [-1, 7, 7, dec_in_channels]
inputs_decoder = 49 * dec_in_channels / 2


def lrelu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))

#%%

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
        mn = tf.layers.dense(x, units=n_latent)
        sd       = 0.5 * tf.layers.dense(x, units=n_latent)            
        epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_latent])) 
        z  = mn + tf.multiply(epsilon, tf.exp(sd))
        
        return z, mn, sd
    
#%%
def decoder(sampled_z, keep_prob):
    with tf.variable_scope("decoder", reuse=None):
        x = tf.layers.dense(sampled_z, 25, activation=lrelu)
        x = tf.layers.dense(x, units=49, activation=lrelu)
        x = tf.reshape(x, reshaped_dim)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
        
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=28*28, activation=None)
        img = tf.reshape(x, shape=[-1, 28, 28])
        return img
    
#%%
sampled, mn, sd = encoder(X_in, keep_prob)
dec = decoder(sampled, keep_prob)

#%%
unreshaped = tf.reshape(dec, [-1, 28*28])
img_loss = tf.reduce_sum(tf.squared_difference(unreshaped, Y_flat), 1)
latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)
loss = tf.reduce_mean(img_loss + kl_factor*latent_loss)
optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)

#%%
sess = tf.Session()


#%%
saver = tf.train.Saver()
save_file='./checkpoints_no_sigmoid_'+str(n_latent)+'_kl_factor_'+str(kl_factor)+'/checkpoints'+str(n_latent)

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
    test_image_stack = (test_image_stack -np.min(test_image_stack))*255/np.max(test_image_stack)
    test_image_stack = np.squeeze(np.round(test_image_stack).astype(np.uint8))
    tiled_output = tile_images(test_image_stack)
    tiled_output = Image.fromarray(tiled_output, mode='L')  # L specifies greyscale
    outfile = os.path.join('./checkpoints_no_sigmoid_'+str(n_latent)+'_kl_factor_'+str(kl_factor), 'epoch_{}.png'.format(epoch))
    tiled_output.save(outfile)


#%%
if save: 
    for i in range(30000):
        batch = [np.reshape(b, [28, 28]) for b in new_batch(train_images, batch_size)]
        sess.run(optimizer, feed_dict = {X_in: batch, Y: batch, keep_prob: 0.8})
            
        if not i % 200:
            saver.save(sess, save_file, global_step=i,write_meta_graph=False)
#            cv2.imwrite('generate_vae_no_sigmoid'+ str(n_latent)+'_step_'+str(i)+'.png', sess.run(dec, feed_dict={sampled:np.random.normal(0,1, (1,n_latent)),keep_prob:1.0})[0,:,:]*256)
            ls, d, i_ls, d_ls, mu, sigm = sess.run([loss, dec, img_loss, latent_loss, mn, sd], feed_dict = {X_in: batch, Y: batch, keep_prob: 1.0})
            print(i, ls, np.mean(i_ls), np.mean(d_ls))
            generate_images(i)        
#%%
