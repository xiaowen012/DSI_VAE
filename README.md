# DSI_VAE
## Description
This repository contains the code and example data used in the paper:
**"Prediction of Fault Slip Tendency in CO₂ Storage using Data-space Inversion."**
\
In this work, we implement a variational autoencoder (VAE)-based data-space inversion (DSI) framework to predict fault slip tendency in CO₂ storage projects. The VAE consists of stacked convolutional long short-term memory (ConvLSTM) layers. The VAE latent variables are then utilized in DSI to generate posterior predictions conditioned on observation data.


## Requirements
- Python 3.9
- TensorFlow 2.10
- numpy
- matplotlib
- scipy
- sklearn


## Contents
- `models/`: Directory to store Python scripts for the VAE (`vae_train.py`) and DSI (`dsi_vae.py`). Additional functions and modules required by these scripts are also included in this directory.
- `quick_testing_data/`: Directory to store example data for quick-testing. This includes training, validation, and testing data saved as `data_train_quicktest.npy`, `data_val_quicktest.npy`, and `data_test_quicktest.npy` files.


## Quick Start
- Download the repository
- Prepare the testing data
The quick_testing_data/ folder includes a minimal working example. No modification is needed.
- Train the Variational Autoencoder (VAE)
Run the following command in the main directory:
```bash
python models/vae_train.py
```
This will load the example dataset, train the VAE model for a few epochs, print the training loss during iterations, and save the trained weights



