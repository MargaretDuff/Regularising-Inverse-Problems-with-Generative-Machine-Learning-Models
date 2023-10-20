# Regularising Inverse Problems with Generative Machine Learning Models
Code for the paper "Regularising Inverse Problems with Generative Machine Learning Models, Margaret Duff, Neill D. F. Campbell, Matthias J. Ehrhardt"

The three folders correspond to the three datasets used in the paper. 


### Requirements 
The code was written using tensorflow v1, you can get round this by installing a version of tensorflow v2 and using the lines

`import tensorflow.compat.v1 as tf`

`tf.compat.v1.disable_v2_behavior()`

The forward models for the inverse problem and the optimisation are written using the [Operator Discretisation Library](https://odlgroup.github.io/odl/). 

## Shapes

The 2d shapes dataset can be generated using the code in `/shapes/datasets/2D-shapes-data.py`. This folder also contains example images. 

To train the three generative models you need wither `ae.py`, `cvae.py` and `wgan.py`.



To get the latent space encodings for the test datasets and to calculate the reconstruction loss, see the file `latent_space.py`. From these encodings different loss measurements on the reconstructions is calculated using the file `diff_loss_measurements.py`. The earth movers distance (EMD) between test and generated distributions is calculated in `run_emd.py` and interpolations created in `run_interpolations.py`. Generating far from the learned distribution is done in `far_from_distribution.py`. Other example reconstructions are shown in the files `main_class_file...`.



## MNIST 

The MNISt data can be downloaded e.g. from https://www.kaggle.com/datasets/hojjatk/mnist-dataset. The notebooks require the training and testing datasets to be in a numpy array: `mnist_train_images.npy` and `mnist_test_images.npy`. 

To train the three generative models you need either `ae_train.py`, `vae_train.py` or `gan_train.py`.

To reconstruct images using the generative models, you want to use the code beginning "main_class_file..". The results can be plotted using the `plot_diagrams.py` file. 

The generative model tests are run in the `generative_tests.py`  and `diff_loss_measurements.py` file. 

## Knees

The main file to run to reconstruct is `run_reconstructions.py`. It requires checkpoints from a trained VAE which can be obtained by running the script `cvae2_12_128_train.py`. The script `plot_diagrams.py` uses `matplotlib` to plot the reconstructed images.  

Example images can be found in the datasets subfolder. The code `datasets/extract-data.py` extracts and reshapes the [fastMRI](https://fastmri.med.nyu.edu/) data according to the paper. 



