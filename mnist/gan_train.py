##https://towardsdatascience.com/teaching-a-variational-autoencoder-vae-to-draw-mnist-characters-978675c95776



# In[3]:

import numpy as np
import sys, getopt
from PIL import Image
import matplotlib.pyplot as plt

latent_dim=8
restore=False
save=True
iteration_start=0
iteration_end=20000

try:
     opts, args = getopt.getopt(sys.argv[1:],"hs:n:r:b:e:d:",["save=","n_latent=", "restore=\
", "iteration_start=", "iteration_end=", "data_set="])
except getopt.GetoptError:
     print('mnist_gan.py -s <True/False> -n <n_latent> -r <"restore file"> -b <iteration_st\
art> -e <iteration_end> ')
     sys.exit(2)
for opt, arg in opts:
     if opt == '-h':
         print('mnist_gan.py -s <True/False> -n <n_latent> -r <"restore file"> -b <iteratio\
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
         restore=str(arg)
         print(restore)
     elif opt in ("-b", "--iteration_start"):
         iteration_start=int(arg)
     elif opt in ("-e", "--iteration_end"):
         iteration_end=int(arg)



# %%
# ## Load the MNIST dataset
# Each MNIST image is originally a vector of 784 integers, each of which is between 0-255 and represents the intensity of a pixel. We model each pixel with a Bernoulli distribution in our model, and we statically binarize the dataset.

train_images=np.load('mnist_train_images.npy')
#test_images=np.load('mnist_test_images.npy')



def new_batch(image_set, batch_size=100):
                numbers=np.random.choice(range(np.shape(image_set)[0]), size=batch_size, replace=False, p=None)
                return image_set[numbers,:]
            

        

# %%
import tensorflow as tf
tf.reset_default_graph()
batch_size = 64

x_true = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='X')
z = tf.placeholder(tf.float32, [None, latent_dim])
keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')

def lrelu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))

#%%

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
    
#%%
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
    
#%%

reconstruction=generator(z, keep_prob)
d_true=discriminator(x_true, keep_prob)
d_generated=discriminator(reconstruction, keep_prob)

with tf.name_scope('regularizer'):
    epsilon = tf.random_uniform([batch_size, 1,1], 0.0, 1.0)
    x_hat = epsilon *  x_true + (1 - epsilon) *  reconstruction
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

#%%
sess = tf.Session()

#%%
saver = tf.train.Saver()
save_file='./checkpoints_ns/checkpoints_gan_'+str(latent_dim)+'/mnist_GAN_'+str(latent_dim)

if restore==False:
    tf.global_variables_initializer().run(session=sess)
else:
    saver=tf.train.Saver()
    saver.restore(sess, restore)

if save:
    sess.run(tf.global_variables_initializer())
    saver.save(sess, save_file, global_step=0, write_meta_graph=True)


#%%
if save: 
    plot_hold_g=np.zeros(iteration_end)
    plot_hold_d=np.zeros(iteration_end)
    for i in range(iteration_start, iteration_end):
        images = new_batch(train_images, batch_size=batch_size)
    
        z_train = np.random.randn(batch_size, latent_dim)
        plot_hold_g[i]=g_loss.eval(session=sess,feed_dict={z: z_train, keep_prob:1.0})
        plot_hold_d[i]=d_loss.eval(session=sess,feed_dict={x_true: images, z: z_train, keep_prob:1.0})
    
        sess.run(g_train, feed_dict={z: z_train, keep_prob: 0.8})
        for j in range(5):
            sess.run(d_train, feed_dict={x_true: images, z: z_train, keep_prob: 0.8})
    
        if i % 100 == 0:
            print('iter={}/{}'.format(i, iteration_end))
            z_validate = np.random.randn(16, latent_dim)
            generated = reconstruction.eval(session=sess,feed_dict={z: z_validate, keep_prob:1.0}).squeeze()
            print('Generator loss is ', g_loss.eval(session=sess,feed_dict={z: z_train, keep_prob:1.0}))    
            print('Discriminator loss is ', d_loss.eval(session=sess,feed_dict={x_true: images, z: z_train, keep_prob:1.0})) 
            saver.save(sess, save_file, global_step=i,write_meta_graph=False)


    plt.plot(range(iteration_end), plot_hold_g)
    plt.title('Generator Loss')
    plt.savefig(save_file+'generator_loss.png')
    plt.plot(range(iteration_end), plot_hold_d)
    plt.title('Discriminator Loss')
    plt.savefig(save_file+'discriminator_loss.png')

   


z_validate = np.random.randn(16, latent_dim)
generated = reconstruction.eval(session=sess, feed_dict={z: z_validate, keep_prob:1.0}).squeeze()
print('generated')
for i in range(16):
    # matplotlib.image.imsave(save_name+'_generated_'+str(i)+'.png', generated[i,:,:])
     Image.fromarray((generated[i,:,:]*255).astype(np.uint8)).save(save_file+'_generated_'+str(i)+'.png')
