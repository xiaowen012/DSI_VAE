import numpy as np
from scipy import linalg
from scipy.stats import norm
from util import *

class parameterization():
    def __init__(self, data, model, is_HT=True, period='HM'):
        self.is_HT = is_HT
        self.period = period
        self.model = model
        self.data = data
        
    def histogramTransformation(self, pred, ori_mean, ori_std, target_sorted):
        for j in range(pred.shape[1]):
            p = norm.cdf(pred[:, j], loc=ori_mean[:, 0], scale=ori_std[:, 0])
            for i in range(p.shape[0]):
                pred[i, j] = quantileSorted(target_sorted[i, :], p[i])
        return pred
    
    def genePredFromKsi(self, z, period='HM'):
        data = self.data
        coeff = self.model.components_ * np.sqrt(np.reshape(self.model.explained_variance_, (-1, 1)))       
        self.period = period
    
        pred = np.dot(coeff.T, z) * self.data['ensemble_std'] + self.data['ensemble_mean']
        
        if self.period == 'HM':       
            if self.is_HT:
                ori_mean = self.data['ensemble_mean'][self.data['obs_time_idx']] 
                ori_std = self.data['ensemble_std'][self.data['obs_time_idx']] 
                target_sorted = self.data['HT_ensemble_sorted'][self.data['obs_time_idx']] 
                pred = self.histogramTransformation(pred[self.data['obs_time_idx']], ori_mean, ori_std, target_sorted)
            pred = pred 
        else:
            if self.is_HT:
                ori_mean = self.data['ensemble_mean']
                ori_std = self.data['ensemble_std']
                target_sorted = self.data['HT_ensemble_sorted'] 
                pred = self.histogramTransformation(pred, ori_mean, ori_std, target_sorted)
            pred = np.maximum(pred, np.zeros(pred.shape))    
        return pred
    
    