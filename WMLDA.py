import numpy as np
from numpy.linalg import inv
from .dpers import DPERm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score


class WLDA(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.mus = None
        self.cov_mat = None
        self.pre_mat = None
        self.pi = None
        self.w = None
        self.G = None
    
    def fit(self, X, y):
        self.G = len(np.unique(y))
        self.mus = np.array([np.nanmean(X[y == g], axis=0) for g in range(self.G)])
        self.cov_mat = DPERm(X, y)
        self.pre_mat = inv(self.cov_mat)
        self.pi = np.array([np.sum(y == g) for g in range(self.G)]) / len(y)
        missing_rate_feature = np.mean(np.isnan(X), axis=0)
        f = lambda r: 1 / (1 - r)
        self.w = np.array([f(r) for r in missing_rate_feature])
        return self
    
    def _classify(self, X):
        pred_label = []
        for i in range(len(X)):
            mask = ((~np.isnan(X[i])).astype(int))
            W = np.diag(mask * self.w)
            temp = np.array([self._log_likelihood(g, i, W, X) for g in range(self.G)])
            pred_label.append(np.argmax(temp))
        return np.array(pred_label)
    
    def _log_likelihood(self, g, i, W, X):
        return np.log(self.pi[g]) - np.matmul((X[i] - self.mus[g]), 
                                              np.matmul(np.matmul(W, np.matmul(self.pre_mat, W)), 
                                              (X[i] - self.mus[g]).T)) / 2

    def predict(self, X):
        return self._classify(X)
    
    def predict_proba(self, X):
        proba = []
        for i in range(len(X)):
            mask = ((~np.isnan(X[i])).astype(int))
            W = np.diag(mask * self.w)
            temp = np.array([self._log_likelihood(g, i, W, X) for g in range(self.G)])
            temp_exp = np.exp(temp - np.max(temp))
            proba.append(temp_exp / temp_exp.sum())
        return np.array(proba)
    
    def score(self, X, y):
        pred_label = self.predict(X)
        return accuracy_score(y, pred_label)