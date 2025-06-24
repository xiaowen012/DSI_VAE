# DSI_VAE
## Description
This repository contains the code and example data used in the paper:
"Prediction of Fault Slip Tendency in CO₂ Storage using Data-space Inversion." In this work, we implement a variational autoencoder (VAE)-based data=space inversion (DSI) framework to predict fault slip tendency in CO₂ storage projects. The VAE consists of stacked convolutional long short-term memory layers. The VAE latent variables are then utilized in DSI.


## Requirements
- TensorFlow 2.10
- Python 3.9
- numpy
- matplotlib
- scipy
- sklearn


## Contents
- `models/`: Directory to store Python scripts for the VAE (vae_train.py) and DSI (dsi_vae.py).
- `quick_testing_data/`: Directory to store example data for quick-testing (including example training, validation, and testing data, saved as 'data_train_quicktest.npy', 'data_val_quicktest.npy' 'data_test_quicktest.npy' files).


## Quick Start
- Download the repository
- Prepare the testing data
\
The quick_testing_data/ folder includes a minimal working dataset. You don’t need to modify anything to run a test.
- Train the Variational Autoencoder (VAE)
Run the following command in the main directory:
\
```bash
python models/vae_train.py
This will:
Load the example dataset,
Train the VAE model for a few epochs,
Print the training loss during iterations,
Save the model outputs

