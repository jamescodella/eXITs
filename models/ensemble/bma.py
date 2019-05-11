from .util import metric
import numpy as np

class BMA():
    def __init__(self):
        super(BMA, self).__init__()

    def fit(self, base_pred, y):
        self.mtr = [0] * base_pred.shape[1] # this is nRMSE for each predictor
        for i in range(base_pred.shape[1]):
            self.mtr[i] = 1. / metric(base_pred[:,i], y)
        # normalize weights
        s = sum(self.mtr)
        for i in range(base_pred.shape[1]):
            self.mtr[i] /= s

    def predict(self, base_pred):
        # prediction based on weighted average
        return np.dot(base_pred, self.mtr)