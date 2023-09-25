##https://towardsdatascience.com/teaching-a-variational-autoencoder-vae-to-draw-mnist-characters-978675c95776



# In[3]:
import tensorflow as tf
import numpy as np
import sys, getopt
alpha=0
beta=0
rho=0.1
n_latent=8
try:
    opts, args = getopt.getopt(sys.argv[1:],"hs:n:a:b:r:",["save=","n_latent=", "alpha=", "beta=", "rho="])
except getopt.GetoptError:
    print('test.py -s <True/False> -n <n_latent>')
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print('test.py -s <True/False> -n <n_latent>')
        sys.exit()
    elif opt in ("-s", "--save"):
        if arg=='True':
            save = True
        else:
            save=False 
    elif opt in ("-n", "--n_latent"):
       n_latent =int( arg)
    elif opt in ("-a", "--alpha"):
        alpha=float(arg)
    elif opt in ("-b", "--beta"):
        beta=float(arg)
    elif opt in ("-r", "--rho"):
        rho=float(arg)

# %%
# ## Load the MNIST dataset
# Each MNIST image is originally a vector of 784 integers, each of which is between 0-255 and represents the intensity of a pixel. We model each pixel with a Bernoulli distribution in our model, and we statically binarize the dataset.

train_images=np.load('mnist_train_images.npy')
test_images=np.load('mnist_test_images.npy')



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
        return mn
    
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
sampled = encoder(X_in, keep_prob)
dec = decoder(sampled, keep_prob)

#%%
def kl_divergence( rho, rho_hat):
        return(rho * tf.log(rho) - rho * tf.log(rho_hat) + (1 - rho) * tf.log(1 - rho) - (1 - rho) * tf.log(1 - rho_hat))
    

unreshaped = tf.reshape(dec, [-1, 28*28])
img_loss = tf.reduce_sum(tf.squared_difference(unreshaped, Y_flat), 1)
sparse_loss =tf.norm(sampled, 1, axis=1)
rho_hat=tf.reduce_mean(sampled,axis=1)
sparse_loss_2=kl_divergence(rho,rho_hat**2)
loss = tf.reduce_mean(img_loss + alpha*sparse_loss+beta*sparse_loss_2)
optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)

#%%
sess = tf.Session()


#%%
saver = tf.train.Saver()
if beta==0:
    save_file='./checkpoints_ns/checkpoints'+str(n_latent)+'/mnist_AE_'+str(n_latent)
    if save:
        sess.run(tf.global_variables_initializer())
        saver.save(sess, save_file, global_step=0, write_meta_graph=True)
    else:
        saver.restore(sess, './checkpoints'+str(n_latent)+'_'+str(alpha)+'/mnist_AE_'+str(n_latent)+'_'+str(alpha)+'-29800')
else:
    save_file='./checkpoints_'+str(n_latent)+'_'+str(beta)+'_'+str(rho)+'/mnist_AE_'+str(n_latent)+'_'+str(alpha)
    if save:
        sess.run(tf.global_variables_initializer())
        saver.save(sess, save_file, global_step=0, write_meta_graph=True)
    else:
        saver.restore(sess, './checkpoints_'+str(n_latent)+'_'+str(beta)+'_'+str(rho)+'/mnist_AE_'+str(n_latent)+'_'+str(alpha)+'-29800')

#%%
if save: 
    for i in range(30000):
        batch = [np.reshape(b, [28, 28]) for b in new_batch(train_images, batch_size)]
        sess.run(optimizer, feed_dict = {X_in: batch, Y: batch, keep_prob: 0.8})
            
        if not i % 200:
            saver.save(sess, save_file, global_step=i,write_meta_graph=False)
                            
            ls, d, i_ls, d_ls, mu = sess.run([loss, dec, img_loss, sparse_loss, sampled], feed_dict = {X_in: batch, Y: batch, keep_prob: 1.0})
            print(i, ls, np.mean(i_ls), np.mean(d_ls))
