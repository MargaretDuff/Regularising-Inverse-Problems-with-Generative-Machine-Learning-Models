# Regularising Inverse Problems with Generative Machine Learning Models
Code for the paper "Regularising Inverse Problems with Generative Machine Learning Models, Margaret Duff, Neill D. F. Campbell, Matthias J. Ehrhardt"

The three folders correspond to the three datasets used in the paper. 


### Requirements 
The code was written using tensorflow v1, you can get round this by installing a version of tensorflow v2 and using the lines

`import tensorflow.compat.v1 as tf`

`tf.compat.v1.disable_v2_behavior()`

The forward models for the inverse problem and the optimisation are written using the [Operator Discretisation Library](https://odlgroup.github.io/odl/). 

## Shapes

## MNIST 

The MNISt data can be o

## Knees

The main file to run to reconstruct is `run_reconstructions.py`. It requires checkpoints from a trained VAE which can be obtained by running the script `cvae2_12_128_train.py`. The script `plot_diagrams.py` uses `matplotlib` to plot the reconstructed images.  

Example images can be found in the datasets subfolder. The code `datasets/extract-data.py` extracts and reshapes the [fastMRI](https://fastmri.med.nyu.edu/) data according to the paper. 



