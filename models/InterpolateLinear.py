import numpy as np
import pickle
import argparse
from sklearn.model_selection import train_test_split
import os
from scipy import interpolate
import argparse
import estimator_pipeline as ep
import data_utils as du

class InterpolateLinear(object):
    def __init__(self, samplesStatsMap, model_count=13):
        super(InterpolateLinear, self).__init__()
        self.model_count = model_count
        self.models = []
        self.samplesStatsMap = samplesStatsMap
        self.__name__ = "InterpolateLinear"

    def getInterpolatedX(self, X, X_time):
        X_int = []
        for subject in range(len(X)):
            pX = X[subject]
            piX = np.zeros(pX.shape)
            for iLabTest in range(piX.shape[0]):
                for jTime in range(piX.shape[1]):
                    chartTimes = X_time[subject][0]
                    nnaPoses = ~np.isnan(pX[iLabTest])
                    nnaPoses[jTime] = False  # pretend that current pos is missing and we want to infer it
                    x = chartTimes[nnaPoses]
                    y = pX[iLabTest, nnaPoses]
                    if jTime == 0 or jTime == piX.shape[1]-1:
                        intFit = interpolate.interp1d(x, y, kind='nearest', fill_value="extrapolate", copy=False, assume_sorted=True)
                    else:
                        intFit = interpolate.interp1d(x, y, kind='linear', fill_value="extrapolate", copy=False, assume_sorted=True)
                    val = intFit([chartTimes[jTime]])[0]
                    val = max(val, self.samplesStatsMap[iLabTest]['min'])
                    val = min(val, self.samplesStatsMap[iLabTest]['max'])
                    piX[iLabTest, jTime] = val
            X_int.append(piX)
        return X_int

    def fill_nans(self, destArr, srcArr, mask=None):
        assert (len(destArr) == len(srcArr))
        for p in range(len(destArr)):
            for iLab in range(destArr[p].shape[0]):
                for jTime in range(destArr[p].shape[1]):
                    if np.isnan(destArr[p][iLab, jTime]):
                        destArr[p][iLab, jTime] = srcArr[p][iLab, jTime]

    def fit(self, X, y, X_mask, y_mask, valid_data, normalizer=None, num_epochs=100, batch_size=64, verbose=1, **kwargs):
        return

    def predict(self, X, X_mask, **kwargs):
        X_original = X.copy()
        X_time = kwargs['X_time']
        X_int = self.getInterpolatedX(X, X_time=X_time)
        self.fill_nans(X_original, X_int, mask=None)
        return X_original

def main(arg):
    if arg.nsplit >= ep.N_SPLITS or arg.nsplit < 0:
        nth_split = ep.N_SPLITS # Using full training data
    else:
        nth_split = arg.nsplit

    pat_idx, X, X_mask, y, y_mask, othr_data = ep.load_ichi_data(data_file='ICHI_training_data.p')
    with open(ep.bam_loc + 'stats.pkl', 'rb') as fp:
        samplesStatsMap = pickle.load(fp)
    estimator = InterpolateLinear(samplesStatsMap)
    mt = ep.ModelTrainer(estimator, X, y, X_mask, y_mask, normalizer=None, **othr_data)
    mt.fit_predict(nth_split=nth_split)     # a value between 0 and 4
    if nth_split == ep.N_SPLITS:
        # Also save the model
        for idx, model in enumerate(mt.estimator_.models):
            mls = ModelLoaderSaver(os.path.join(ep.output_models, os.path.basename(__file__).rstrip('.py') + "_f" + str(idx)))
            mls.save_standard_model(model)

def main_load_models(arg):
    with open(ep.bam_loc + 'stats.pkl', 'rb') as fp:
        samplesStatsMap = pickle.load(fp)
    estimator = InterpolateLinear(samplesStatsMap)    # load model
    pat_idx, X, X_mask, othr_data = ep.load_ichi_test_data(data_file=ep.TEST_DATA_file,is_padded=False)
    mt = ep.ModelTester(estimator, X, X_mask, normalizer=None, **othr_data)
    mt.predict()

if __name__ == "__main__":
    arg = du.parse_args()
    if arg.final:
        main_load_models(arg)
    else:
        main(arg)