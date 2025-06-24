# DSI_VAE
## Description
This repository contains the code and example data used in the paper:
"Prediction of Fault Slip Tendency in CO₂ Storage using Data-space Inversion." In this work, we implement a variational autoencoder (VAE)-based data=space inversion (DSI) framework to predict fault slip tendency in CO₂ storage projects. The VAE consists of stacked convolutional long short-term memory layers. The VAE latent variables are then utilized in DSI.


## Requirements
- TensorFlow 2.10.0
- Python 3.9
- numpy
- matplotlib
- scipy
- sklearn


## Usage
- `models/`: Directory to store Python scripts for VAE and DSI.
- `quick_testing_data/`: Directory to store example data for quick-testing. 

To run a quick test, 
```bash
python models/vae_train.py
