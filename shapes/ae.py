import numpy as np
import sys
import getopt
import tensorflow as tf

#Set arguments
n_latent = 12
dec_in_channels = 1
restore = False
save = True
iteration_start = 0
iteration_end = 30000
base = '../datasets/2d_shapes/2d_shapes_train_images_'
name = 'none'
try:
    opts, args = getopt.getopt(sys.argv[1:], "hs:n:r:b:e:d:", [
                               "save=", "n_latent=", "restore=", "iteration_start=", "iteration_end=", "data_set="])
except getopt.GetoptError:
    print('2d_shapes_ae.py -s <True/False> -n <n_latent> -r <"restore file"> -b <iteration_start> -e <iteration_end> -d <none/circle/rect>')
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print('2d_shapes_ae.py -s <True/False> -n <n_latent> -r <"restore file"> -b <iteration_start> -e <iteration_end>')
        sys.exit()
    elif opt in ("-s", "--save"):
        if arg == 'True':
            save = True
        else:
            save = False
    elif opt in ("-n", "--n_latent"):
        n_latent = int(arg)
    elif opt in ("-r", "--restore"):
        restore = str(arg)
    elif opt in ("-b", "--iteration_start"):
        iteration_start = int(arg)
    elif opt in ("-e", "--iteration_end"):
        iteration_end = int(arg)
    elif opt in ("-d", "--data_set"):
        name = str(arg)
        data_set = base+str(arg)+'.npy'



#Load dataset and sort batching
batch_size = 32
data_set = base+'none.npy'
train_images = np.load(data_set)
def new_batch(image_set, batch_size=100):
    numbers = np.random.choice(
        range(np.shape(image_set)[0]), size=batch_size, replace=False, p=None)
    return image_set[numbers, :]




#Build the AE 
tf.reset_default_graph()
sess = tf.InteractiveSession()


X_in = tf.placeholder(dtype=tf.float32, shape=[None, 56, 56], name='X')
Y = tf.placeholder(dtype=tf.float32, shape=[None, 56, 56], name='Y')
Y_flat = tf.reshape(Y, shape=[-1, 56 * 56])
keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')
reshaped_dim = [-1, 7, 7, dec_in_channels]

def lrelu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))

def encoder(X_in, keep_prob=1.0):
    activation = lrelu
    with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
        X = tf.reshape(X_in, shape=[-1, 56, 56, 1])
        x = tf.layers.conv2d(X, filters=64, kernel_size=4,
                             strides=1, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4,
                             strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4,
                             strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.contrib.layers.flatten(x)
        mn = tf.layers.dense(x, units=n_latent)
        return mn


def decoder(sampled_z, keep_prob=1.0):
    activation = lrelu
    with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
        x = tf.layers.dense(sampled_z, units=49, activation=lrelu)
        x = tf.reshape(x, reshaped_dim)
        x = tf.layers.conv2d_transpose(
            x, filters=64, kernel_size=4, strides=2, padding='same', activation=lrelu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(
            x, filters=64, kernel_size=4, strides=2, padding='same', activation=lrelu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(
            x, filters=64, kernel_size=4, strides=1, padding='same', activation=lrelu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=56*56, activation=None)
        img = tf.reshape(x, shape=[-1, 56, 56])
        return img


sampled = encoder(X_in, keep_prob)
dec = decoder(sampled, keep_prob)

#Set up loss function and optimiser
unreshaped = tf.reshape(dec, [-1, 56*56])
img_loss = tf.reduce_sum(tf.squared_difference(unreshaped, Y_flat), 1)
loss = tf.reduce_mean(img_loss)
optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)


#Set up loading and saving of checkpoints 
sess = tf.Session()
saver = tf.train.Saver()
save_file = './checkpoints_no_sigmoid/checkpoints_' + \
    name+'_'+str(n_latent)+'/2d_shapes_AE'
if save:
    sess.run(tf.global_variables_initializer())
    saver.save(sess, save_file, global_step=0, write_meta_graph=True)
else:
    saver.restore(sess, './checkpoints_no_sigmoid/checkpoints_' +
                  name+'_'+str(n_latent)+'/2d_shapes_AE-29600')

#Training loop 
if save:
    for i in range(iteration_start, iteration_end):
        batch = [np.reshape(b, [56, 56])
                 for b in new_batch(train_images, batch_size)]
        sess.run(optimizer, feed_dict={X_in: batch, Y: batch, keep_prob: 0.8})

        if not i % 200:
            saver.save(sess, save_file, global_step=i, write_meta_graph=False)

            ls, d, i_ls = sess.run([loss, dec, img_loss], feed_dict={
                                   X_in: batch, Y: batch, keep_prob: 1.0})
            print(i, ls, np.mean(i_ls))
