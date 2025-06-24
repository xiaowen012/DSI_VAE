import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import h5py
from layers_3D import *
from keras import backend as K
from keras.layers import Input, Flatten, Dense, Lambda, Reshape, concatenate, TimeDistributed, RepeatVector, ConvLSTM3D, ConvLSTM2D
from keras.models import Model
from keras.optimizers import Adam
from keras import layers, models
from keras import backend as K
from keras.layers import Lambda
from keras.losses import mse

plt.set_cmap('jet')
plt.rcParams['font.size'] = '16'


# Load quick-test datasets (train/val/test)
base_dir = os.path.dirname(os.path.abspath(__file__))
data_train = np.load(os.path.join(base_dir, '../quick_testing_data/data_train_quicktest.npy'))
data_val = np.load(os.path.join(base_dir, '../quick_testing_data/data_val_quicktest.npy'))
data_test = np.load(os.path.join(base_dir, '../quick_testing_data/data_test_quicktest.npy'))


# Sampling function for VAE
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1:]
    epsilon = K.random_normal(shape=(batch,) + dim)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


# Define VAE model
def create_vae(input_dim, latent_dim):
    # --- Encoder ---
    encoder_input = Input(shape=input_dim)
    x = ConvLSTM3D(16, (3, 3, 3), strides=(2, 2, 2), padding = 'same', activation='relu', return_sequences = True)(encoder_input)
    x = ConvLSTM3D(32, (3, 3, 3), strides=(2, 2, 2), padding = 'same', activation='relu', return_sequences = True)(x)
    x = ConvLSTM3D(64, (3, 3, 3), strides=(2, 2, 2), padding = 'same', activation='relu', return_sequences = True)(x)
    z = ConvLSTM3D(16, (3, 3, 3), strides=(1, 1, 1), padding = 'same', activation = None, return_sequences = False)(x)  
    z_flatten = Flatten()(z)
    z_mean = Dense(latent_dim)(z_flatten)
    z_log_var = Dense(latent_dim)(z_flatten)
    z_sample = Lambda(sampling, output_shape=latent_dim)([z_mean, z_log_var])
    
    encoder = Model(encoder_input, [z_mean, z_log_var, z_sample], name='encoder')
    
    # --- Decoder ---
    decoder_input = Input(shape = latent_dim)
    x = Dense(3 * 7 * 7 * 16)(decoder_input)
    x = Reshape((3, 7, 7, 16))(x)
    x = RepeatConv(n_t)(x)
    x = ConvLSTM3D(16, (3, 3, 3), strides=(1, 1, 1), padding = 'same', activation='relu', return_sequences = True)(x) 
    x = ConvLSTM3D(64, (3, 3, 3), strides=(1, 1, 1), padding = 'same', activation='relu', return_sequences = True)(x)
    x = time_dconv_bn_nolinear(64, 3, 3, 3, stride=(2, 2, 2))(x)
    x = ConvLSTM3D(128, (3, 3, 3), strides=(1, 1, 1), padding = 'same', activation='relu', return_sequences = True)(x)
    x = time_dconv_bn_nolinear(128, 3, 3, 3, stride=(2, 2, 2))(x)
    x = time_cropping_3D(cropping = (1, 1, 1))(x)    
    x = ConvLSTM3D(32, (3, 3, 3), strides=(1, 1, 1), padding = 'same', activation='relu', return_sequences = True)(x)
    x = time_dconv_bn_nolinear(32, 3, 3, 3, stride=(2, 2, 2))(x)
    x = time_cropping_3D(cropping = (0, 1, 1))(x)
    decoder_output = ConvLSTM3D(4, (3, 3, 3), strides=(1, 1, 1), padding = 'same', activation = None, return_sequences = True)(x)   
    
    decoder = Model(decoder_input, decoder_output, name = 'decoder')

    return encoder, decoder


# Model configuration
n_t = 7
n_z = 20
input_dim=(n_t, n_z, 50, 50, 4)  # full input shape
latent_dim  = 128
encoder, decoder = create_vae(input_dim, latent_dim)

input_encoder = Input(shape = input_dim)
[z_mean, z_log_var, z_latent] = encoder(input_encoder)
decoder_output = decoder(z_latent)
vae_model = Model(input_encoder, decoder_output)


# Define reconstruction losses (evaluate only on selected time steps)
selected_time = [0, 1, 2, 3, 6]
selected_input_encoder = tf.gather(input_encoder, [0, 1, 2, 3, 6], axis=1)
selected_decoder_output = tf.gather(decoder_output, [0, 1, 2, 3, 6], axis=1)

reconstruction_loss_p = tf.reduce_mean(tf.square(selected_input_encoder[..., 0:1] - selected_decoder_output[..., 0:1]))
reconstruction_loss_s = tf.reduce_mean(tf.square(selected_input_encoder[..., 1:2] - selected_decoder_output[..., 1:2]))

region1_input = selected_input_encoder[:, :, :, 16, 10:50, 2:3]
region1_output = selected_decoder_output[:, :, :, 16, 10:50, 2:3]
region2_input = selected_input_encoder[:, :, :, 32, :40, 2:3]
region2_output = selected_decoder_output[:, :, :, 32, :40, 2:3]
region1_loss = tf.reduce_mean(tf.square(region1_input - region1_output))
region2_loss = tf.reduce_mean(tf.square(region2_input - region2_output))
reconstruction_loss_normalstress = region1_loss + region2_loss
region1_input = selected_input_encoder[:, :, :, 16, 10:50, 3:4]
region1_output = selected_decoder_output[:, :, :, 16, 10:50, 3:4]
region2_input = selected_input_encoder[:, :, :, 32, :40, 3:4]
region2_output = selected_decoder_output[:, :, :, 32, :40, 3:4]
region1_loss = tf.reduce_mean(tf.square(region1_input - region1_output))
region2_loss = tf.reduce_mean(tf.square(region2_input - region2_output))
reconstruction_loss_shearstress = region1_loss + region2_loss

# Total reconstruction loss
reconstruction_loss = reconstruction_loss_p + reconstruction_loss_s + reconstruction_loss_normalstress + reconstruction_loss_shearstress

# KL divergence loss
kl = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
kl_loss = tf.reduce_mean(kl)
# Full VAE loss
vae_loss = 1e4 * reconstruction_loss + kl_loss

# Track additional metrics for monitoring
vae_model.add_loss(vae_loss)
vae_model.add_metric(reconstruction_loss, name='reconstruction_loss', aggregation='mean')
vae_model.add_metric(kl_loss, name='kl_loss', aggregation='mean')
vae_model.add_metric(reconstruction_loss_p, name='reconstruction_loss_p', aggregation='mean')
vae_model.add_metric(reconstruction_loss_s, name='reconstruction_loss_s', aggregation='mean')
vae_model.add_metric(reconstruction_loss_normalstress, name='reconstruction_loss_normalstress', aggregation='mean')
vae_model.add_metric(reconstruction_loss_shearstress, name='reconstruction_loss_shearstress', aggregation='mean')

# Compile model
opt = Adam(learning_rate=7.5e-04)
vae_model.compile(optimizer=opt)

# Define training callbacks
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
lrScheduler = ReduceLROnPlateau(monitor = 'loss', factor = 0.8, patience = 5, cooldown = 1, verbose = 1, min_lr = 1e-7)
filePath = './vae_weights/saved-model-{epoch:03d}-{val_loss:.2f}.h5'
checkPoint = ModelCheckpoint(filePath, monitor = 'val_loss', verbose = 1, save_best_only = False, \
                             save_weights_only = True, mode = 'auto', period = 5)

callbacks_list = [lrScheduler, checkPoint]


# Start training
epochs = 10
batch_size = 8
history = vae_model.fit(data_train, data_train, batch_size = batch_size, epochs = epochs, \
                        verbose = 1, validation_data = (data_val, data_val), callbacks = callbacks_list, shuffle=True)
