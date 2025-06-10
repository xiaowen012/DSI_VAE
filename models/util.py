import numpy as np
from scipy import linalg
from scipy.stats import norm

def quantileSorted(valsSorted, p):
    '''
    quantile on sorted vals (vals: ascending order)
    '''
    numReal = valsSorted.shape[0]
    tmpIdx = numReal * p + 0.5
    flIdx = tmpIdx.astype(int)
    if flIdx <= 0:
        val = valsSorted[0]
    elif flIdx == numReal:
        val = valsSorted[numReal-1]
    else:
        val = (tmpIdx - flIdx) * valsSorted[flIdx] + (flIdx + 1 - tmpIdx) * valsSorted[flIdx-1]
    return val

def ecdfCalculation(ensemble, data):
    '''
    ensemble: the data ensemble and show the distribution /sorted data / matrix: num_timestep * num_samples
    data: the target data to generate the cdf
    '''
    ecdf = np.zeros(data.shape)
    num_sum = ensemble.shape[1]
    for i in range(data.shape[0]):  
        idx = np.interp(data[i], ensemble[i, ...], range(num_sum))
        ecdf[i] = idx / num_sum       
    return ecdf



def preprocssToMatrixFormat(ensemble, obs_time_idx):
    data = {}
    n_qoi, n_step, n_real = ensemble.shape
    data['val'] = ensemble
    data['ensemble'] = np.reshape(ensemble, (n_qoi * n_step, n_real))
    
    data['ensemble_mean'] = np.expand_dims(np.mean(data['ensemble'], axis=1), axis=1)
    data['ensemble_std'] = np.expand_dims(np.maximum(np.std(data['ensemble'], axis=1), 1e-6), axis=1) 
    data['ensemble_centered'] = (data['ensemble'] - data['ensemble_mean']) / data['ensemble_std']
    data['ensemble_centered_mean'] = np.expand_dims(np.mean(data['ensemble_centered'], axis=1), axis=1)
    data['ensemble_centered_std'] = np.expand_dims(np.maximum(np.std(data['ensemble_centered'], axis=1), 1e-6), axis=1)
                                                   

    data['ensembleObsTsteps'] = data['val'][:, obs_time_idx, :].reshape((-1, n_real))
    data['ensembleObsTsteps_mean'] = np.expand_dims(np.mean(data['ensembleObsTsteps'], axis=1), axis=1)    
    
    data['HT_simHist_mean'] = data['ensembleObsTsteps_mean']
    data['HT_simHist_std'] = np.expand_dims(np.maximum(np.std(data['ensembleObsTsteps'], axis=1), 1e-6), axis=1)
    data['HT_simHist_sorted'] = np.sort(data['ensembleObsTsteps']) # all * nsamples
    data['HT_ensemble_sorted'] = np.sort(data['ensemble']) # all * nsamples
    data['obs_time_idx'] = obs_time_idx
    return data


def cal_matrix_inverse(cd):
    varn = 0.95
    u, s, vh = linalg.svd(cd); 
    v = vh.T
    diagonal = s
    for i in range(len(diagonal)):
        if (sum(diagonal[0:i+1]))/(sum(diagonal)) > varn:
            diagonal = diagonal[0:i+1]
            break
    u=u[:,0:i+1]
    v=v[:,0:i+1]
    ss = np.diag(diagonal**(-1))
    cd_inv = np.dot(np.dot(v,ss),u.T)
    return cd_inv

def cal_mahalanobis_dist(d, mu, cd_inv):
    dist = np.sqrt(np.dot(np.dot((d-mu).T, cd_inv), (d - mu)))
    return dist


    
def relativeErrorTimeAverage(true_data, prediction):
    error = np.sum(np.absolute(true_data[:, :, :] - prediction[:, :, :]), axis=2) / np.sum(true_data[:, :, :], axis=2)
    error = np.mean(error, axis=1)
    return error