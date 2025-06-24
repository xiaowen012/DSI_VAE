import numpy as np
from scipy import array, linalg, dot
from sklearn.decomposition import PCA

class ESMDA:
    def __init__(self, nm=1, nd=1, nr=1, sim_master=None):
        
        self.nm = nm  # number of model parameters
        self.nd = nd  # number of data
        self.nr = nr  # number of realizations (for single time I supposed)
        self.d_obs = None  #(nd, 1) observed data
        self.d_uc = np.zeros((self.nd, self.nr)) #(nd, nr)  observed data
        self.m_prior = None #np.zeros((nm, nr))  # prior models
        self.m_posterior = None #np.zeros((nm, nr))  # posterior models
        self.d_posterior = None #np.zeros((nd, nr))  # posterior predictions
        self.m_k = np.zeros((self.nm, self.nr)) #(nm, nr)  # models at k-th iteration
        self.d_k = np.zeros((self.nd, self.nr)) #(nd, nr)  # prediction at k-th iteration
        self.cd = None #(nd, nd)  # covariance of data error
        self.cm = None #(nm, nm)  # covariance of model parameters
        
        self.cmd = None #(self.nm, self.nd)
        self.cdd = None #(self.nd, self.nd)
        self.dn = None #(self.nd, self.nr)
        self.na = 4
        self.i_na = 0
        self.alpha = np.ones((self.na, 1)) * self.na
        
        self.sim_master = sim_master  # a class for performing simulation  
        
    def initialize(self):
        # Initialize the model ensemble for assimilation
        self.m_k = np.copy(self.m_prior)

    def perturb_observation(self):
        # Perturb the observed data with multivariate Gaussian noise
        cd = self.cd
        for i in range(0, self.nr):      
            self.d_uc[:, i] = (self.d_obs + np.random.multivariate_normal(np.zeros(self.nd), self.alpha[self.i_na] * cd, 1).T).reshape((-1, ))

    def forecast(self):
        # Forecast step
        self.d_k = self.sim_master.genePredFromKsi(self.m_k)
        return True

    def update(self):
        # Update based on perturbed observations using ESMDA
        self.perturb_observation()
        m_ave = np.mean(self.m_k, 1, keepdims=True)
        d_ave = np.mean(self.d_k, 1, keepdims=True)
        ones = np.ones((1, self.nr))
        self.cmd = np.dot(self.m_k - np.dot(m_ave, ones), 
                          np.transpose(self.d_k - np.dot(d_ave, ones))) / (self.nr - 1)
        self.cdd = np.dot(self.d_k - np.dot(d_ave, ones), 
                          np.transpose(self.d_k - np.dot(d_ave, ones))) / (self.nr - 1)
        
        cd_inv = np.linalg.inv(self.cdd + self.alpha[self.i_na] * self.cd)
        
        for i in range(0, self.nr):
            self.m_k[:, i] = np.copy(self.m_k[:, i]) + np.dot(np.dot(self.cmd, cd_inv), self.d_uc[:, i] - self.d_k[:, i])

        self.i_na += 1

    def solve(self):
        self.initialize()
        while self.i_na < self.na:
            self.forecast()
            self.update()
        self.save_result()

    def save_result(self):
        self.d_posterior = np.copy(self.d_k)
        self.m_posterior = np.copy(self.m_k)
        # save the posterior results 
        np.save('esmda_result/m_posterior.npy', self.m_posterior)
        np.save('esmda_result/d_posterior.npy', self.d_posterior)


    def input_m_prior(self, m_prior):
        self.m_prior = m_prior
        if self.m_prior.shape[0] != self.nm or self.m_prior.shape[1] != self.nr:
            print("[ERROR] Dimension mismatch for m_prior.")
            exit(1)

    def input_d_obs(self, d_obs):
        self.d_obs = d_obs
        if self.d_obs.shape[0] != self.nd:
            print("[ERROR] Dimension mismatch for d_obs.")
            exit(1)


    def input_d_uc(self, d_uc):
        self.d_uc = d_uc
        if self.d_uc.shape[0] != self.nd or self.d_uc.shape[1] != self.nr:
            print("[ERROR] Dimension mismatch for d_uc.")
            exit(1)

    def input_cd(self, cd):
        self.cd = cd
        if self.cd.shape[0] != self.nd or self.cd.shape[1] != self.nd:
            print("[ERROR] Dimension mismatch for C_d.")
            exit(1)  
    
    def input_cm(self, cm):
        self.cm = cm
        if self.cm.shape[0] != self.nm or self.cm.shape[1] != self.nm:
            print("[ERROR] Dimension mismatch for C_m.")
            exit(1)

    def input_na(self, na):
        self.na = na
        if self.na < 1:
            print('[ERROR] Invalid Na value')
            exit(1)

    def input_alpha(self, alpha):
        self.alpha = alpha
        if np.abs(np.sum(1/self.alpha) - 1) > 1e-3:
            print('[ERROR] Sum of multiplication coefficients not equal to 1.')
            exit(1)
