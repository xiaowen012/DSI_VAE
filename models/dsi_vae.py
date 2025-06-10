import numpy as np
import scipy.io

from util import *
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from sklearn import linear_model
from scipy.stats import norm
from ESMDA import ESMDA
from scipy import array, linalg, dot
from scipy import interpolate
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt

plt.set_cmap('jet')
plt.rcParams['font.size'] = '16'

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
from tensorflow import keras
import h5py
from layers_3D import *
from keras import backend as K
from keras.layers import Input, Flatten, Dense, Lambda, Reshape, concatenate, TimeDistributed, RepeatVector, ConvLSTM3D, ConvLSTM2D
from keras.models import Model
from keras.optimizers import Adam


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1:]
    epsilon = K.random_normal(shape=(batch,) + dim)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def create_vae(input_dim, latent_dim):

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


n_t = 7
n_z = 20
input_dim=(n_t, n_z, 50, 50, 4)
latent_dim  = 256
encoder, decoder = create_vae(input_dim, latent_dim)

input_encoder = Input(shape = input_dim)
[z_mean, z_log_var, z_latent] = encoder(input_encoder)
decoder_output = decoder(z_latent)
vae_model = Model(input_encoder, decoder_output)

vae_model.load_weights('saved-model.h5')
vae_model.summary()


data_train = np.load('data_train_norm.npy')
data_eval = np.load('data_eval_norm.npy')
data_test = np.load('data_test_norm.npy')
data_all = np.concatenate([data_train, data_eval, data_test], axis=0)


[_, _, z_full] = encoder.predict(data_all)
num_train = 1200


e_field = np.load('params/e_field.npy')
perm_multiplier1 = np.load('params/perm_multiplier1.npy')
perm_multiplier2 = np.load('params/perm_multiplier2.npy')
poisson = np.load('params/poisson.npy')
biot = np.load('params/biot.npy')
gamma = np.load('params/gamma.npy')


max_e_field = np.max(e_field[:num_train, :])
min_e_field = np.min(e_field[:num_train, :])
e_field_norm = (e_field - min_e_field) / (max_e_field - min_e_field)
print(max_e_field, min_e_field, np.min(e_field_norm), np.max(e_field_norm))

max_perm_multiplier1 = np.max(perm_multiplier1[:num_train, :])
min_perm_multiplier1 = np.min(perm_multiplier1[:num_train, :])
perm_multiplier1_norm = (perm_multiplier1 - min_perm_multiplier1) / (max_perm_multiplier1 - min_perm_multiplier1)
print(max_perm_multiplier1, min_perm_multiplier1, np.min(perm_multiplier1_norm), np.max(perm_multiplier1_norm))

max_perm_multiplier2 = np.max(perm_multiplier2[:num_train, :])
min_perm_multiplier2 = np.min(perm_multiplier2[:num_train, :])
perm_multiplier2_norm = (perm_multiplier2 - min_perm_multiplier2) / (max_perm_multiplier2 - min_perm_multiplier2)
print(max_perm_multiplier2, min_perm_multiplier2, np.min(perm_multiplier2_norm), np.max(perm_multiplier2_norm))

max_poisson = np.max(poisson[:num_train, :])
min_poisson = np.min(poisson[:num_train, :])
poisson_norm = (poisson - min_poisson) / (max_poisson - min_poisson)
print(max_poisson, min_poisson, np.min(poisson_norm), np.max(poisson_norm))

max_biot = np.max(biot[:num_train, :])
min_biot = np.min(biot[:num_train, :])
biot_norm = (biot - min_biot) / (max_biot - min_biot)
print(max_biot, min_biot, np.min(biot_norm), np.max(biot_norm))

max_gamma = np.max(gamma[:num_train, :])
min_gamma = np.min(gamma[:num_train, :])
gamma_norm = (gamma - min_gamma) / (max_gamma - min_gamma)
print(max_gamma, min_gamma, np.min(gamma_norm), np.max(gamma_norm))


y_all = np.concatenate([e_field_norm, perm_multiplier1_norm, perm_multiplier2_norm, 
                        poisson_norm, biot_norm, gamma_norm], axis = -1)
print(y_all.shape)
y_min = np.min(y_all, axis = 0, keepdims=True)
y_max = np.max(y_all, axis = 0, keepdims=True)


wells = [25, 25, 29, 34, 19, 22, 34, 15]
well_loc_x = wells[:4]
well_loc_y = wells[4:]

p_mean, p_std = 18465360.0, 1516672.1
strain_mean, strain_std = 8.40719e-05, 7.3231065e-05
ns_mean, ns_std = 7375302.27780193, 1026706.3068824525
ss_mean, ss_std = 3675581.8832918517, 486383.079628042

pressure_data = data_all[..., 0] * p_std + p_mean
strain_data = data_all[..., 1] * strain_std + strain_mean

n_r = pressure_data.shape[0]

n_layer_select_p = np.array(range(20))
n_layer_select_s = np.array(range(20))
n_t_select = [-1] 
n_t_list = [0, 1, 2, 3] 
n_t_hm = len(n_t_list)

dp = pressure_data[:n_r, n_t_list][:, :, n_layer_select_p][...,well_loc_y, well_loc_x].reshape((n_r, -1)) * 1e-6
ds = strain_data[:n_r, n_t_list][:, :, n_layer_select_s][...,well_loc_y, well_loc_x].reshape((n_r, -1))

error_p = 0.1
error_s = 1e-5

n_well = len(well_loc_x)
n_hm_p = n_well * len(n_layer_select_p) * n_t_hm
n_hm_s = n_well * len(n_layer_select_s) * n_t_hm
n_hm = n_hm_p + n_hm_s
obs_data_std = np.concatenate([error_p * np.ones((n_hm_p,)), error_s * np.ones((n_hm_s,))], axis = 0)
cd = np.diag(obs_data_std ** 2)


class autoencoder_parameterization():
    def __init__(self, period='HM'):
        self.period = period
    def genePredFromKsi(self, z, period='HM'):
        z = z.T
        z = z[:, :-6]
        
        pred_dsi = decoder.predict(z)
        pressure = pred_dsi[..., 0][..., np.newaxis]
        strain = pred_dsi[..., 1][..., np.newaxis]
        normal_stresses = pred_dsi[..., 2][..., np.newaxis]
        shear_stresses = pred_dsi[..., 3][..., np.newaxis]

        pressure = (pressure[..., 0] * p_std + p_mean) * 1e-6
        strain = strain[..., 0] * strain_std + strain_mean 
        normal_stresses = normal_stresses[..., 0] * ns_std + ns_mean
        shear_stresses = shear_stresses[..., 0] * ss_std + ss_mean
        
        pred = np.concatenate([pressure[..., np.newaxis], strain[..., np.newaxis], normal_stresses[..., np.newaxis], shear_stresses[..., np.newaxis]], -1)
        if period == 'HM':
            pred_pressure = pred[:, range(4)][:, :, n_layer_select_p][..., well_loc_y, well_loc_x, 0].reshape((pred.shape[0], -1))
            pred_strain = pred[:, range(4)][:, :, n_layer_select_s][..., well_loc_y, well_loc_x, 1].reshape((pred.shape[0], -1))
            pred = np.concatenate([pred_pressure, pred_strain], axis=-1)
            pred = pred.T
        return pred


dhm_ensemble = np.concatenate([dp, ds], axis=-1)
df_ensemble = dhm_ensemble
dfull_ensemble = dhm_ensemble
df_ensemble = df_ensemble.astype('float32')
dfull_ensemble = dfull_ensemble.astype('float32')
dhm_ensemble = dhm_ensemble.astype('float32')
print(dhm_ensemble.shape, df_ensemble.shape, dfull_ensemble.shape)
print(np.max(dhm_ensemble), np.min(dhm_ensemble), np.max(ds), np.min(ds))

obs_time_idx = range(n_hm)
n_obs = n_hm
n_obs_p = n_hm_p


i_list = [1350]

for true_idx in i_list:
    true_data = dfull_ensemble[true_idx]
    obs_data_para = {}
    rand_num =  np.random.randn(n_obs)
    obs_data_val = true_data.copy()[obs_time_idx]
    obs_data_val[:n_obs_p] = true_data[obs_time_idx][:n_obs_p] + error_p * rand_num[:n_obs_p]
    obs_data_val[n_obs_p:] = true_data[obs_time_idx][n_obs_p:] + error_s * rand_num[n_obs_p:]
  
    obs_data_std = obs_data_val.copy()
    obs_data_std[:n_obs_p] = error_p
    obs_data_std[n_obs_p:] = error_s
    
    obs_data_std_in_likeli = obs_data_std

    obs_data_para['val'] = obs_data_val
    obs_data_para['std'] = obs_data_std
    obs_data_para['std_in_likeli'] = obs_data_std_in_likeli

    is_HT = True 
    nd = n_hm
    nr = 600
    nr_list = range(0, 1200, 2)

    na = 4   
    alpha = np.array([9.333, 7.0, 4.0, 2.0])
    d_obs = obs_data_para['val'].reshape((-1, 1))    
    cd_list = obs_data_para['std_in_likeli'] ** 2
    cd = np.diag(cd_list.reshape((-1, )))
    
    ae_para = autoencoder_parameterization()
    nm = 256 + 6
    print('nm: ', nm)
    m_prior = np.concatenate([z_full[nr_list], y_all[nr_list]], axis=-1).T  
    print(m_prior.shape)
    es_mda = ESMDA(nm, nd, nr, ae_para)
    es_mda.input_m_prior(m_prior)
    es_mda.input_d_obs(d_obs)
    es_mda.input_cd(cd)
    es_mda.input_na(na)
    es_mda.input_alpha(alpha)

    print('Begin solve.')
    es_mda.solve()
    pred_ae_post = ae_para.genePredFromKsi(es_mda.m_posterior, 'full')
    print('Realization ', true_idx, ': End posterior prediction')
    print(pred_ae_post.shape)

    # save results
    pred_results = {}
    pred_results['post_d_aae'] = pred_ae_post
    pred_results['post_m_aae'] = es_mda.m_posterior
    
    np.save('./vae_jointparam/'+str(true_idx) +'_pred.npy', pred_results)
    np.save('./vae_jointparam/'+str(true_idx) +'_obs.npy', obs_data_para)


