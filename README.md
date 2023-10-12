# Regularising Inverse Problems with Generative Machine Learning Models
Code for the paper "Regularising Inverse Problems with Generative Machine Learning Models, Margaret Duff, Neill D. F. Campbell, Matthias J. Ehrhardt"

The three folders correspond to the three datasets used in the paper. 

## Shapes

## MNIST 

## Knees

The main file to run to reconstruct is `run_reconstructions.py`. It requires checkpoints from a trained VAE which can be obtained by running the script `cvae2_12_128_train.py`. The script `plot_diagrams.py` uses `matplotlib` to plot the reconstructed images.  

Example images can be found in the datasets subfolder. The code `datasets/extract-data.py` extracts and reshapes the [fastMRI](https://fastmri.med.nyu.edu/) data according to the paper. 



