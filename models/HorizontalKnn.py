import numpy as np
import pickle
import argparse
from sklearn.model_selection import train_test_split
import os
import estimator_pipeline as ep
from ensemble.util import ModelLoaderSaver
from sklearn.neighbors import KNeighborsRegressor
import copy
import data_utils as du
from HorizontalBase import HorizontalBase

class HorizontalKnn(HorizontalBase):
    def __init__(self, samplesStatsMap, model_count=13):
        super(HorizontalKnn, self).__init__("HorizontalKnn", samplesStatsMap, model_count)
        self.models = []
        for i in range(model_count):
            model = KNeighborsRegressor( n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
            self.models.append(model)

    def fit(self, X, y, X_mask, y_mask, valid_data, normalizer=None, num_epochs=100, batch_size=64, verbose=1,
            **kwargs):
        X_int = kwargs['X_interpolated_linear']
        X = self.normalizeZScore(X)
        X_int = self.normalizeZScore(X_int)
        X_flat = np.hstack(X)
        X_int_flat = np.hstack(X_int)
        y_flat = np.hstack(y)
        print('Training')
        for iLab in range(self.model_count):
            print('\t' + str(iLab + 1) + '/' + str(self.model_count))
            X_col, y_col = self.prepareXY(X_flat, X_int_flat, y_flat, iLab, remove_y_nan=True)
            self.models[iLab].fit(X_col, y_col)

    def predict(self, X, X_mask, **kwargs):
        X_original = X.copy()
        X = self.normalizeZScore(X.copy())
        X_int = kwargs['X_interpolated_linear']
        X_int = self.normalizeZScore(X_int)
        X_flat = np.hstack(X)
        X_int_flat = np.hstack(X_int)
        samplesPerSubject = [px.shape[1] for px in X]
        y_pred_flat = []
        print('Testing')
        for iLab in range(self.model_count):
            print('\t' + str(iLab+1) + '/' + str(self.model_count))
            X_col, y_col = self.prepareXY(X_flat, X_int_flat, X_int_flat, iLab, remove_y_nan=False)
            y_pred_p = self.models[iLab].predict(X_col)
            y_pred_flat.append(y_pred_p)

        y_pred_flat = np.vstack(y_pred_flat).transpose()
        y_pred = self.splitIntoParticipants(y_pred_flat, samplesPerSubject)
        self.fill_nans(X_original, y_pred, mask=None)
        return X_original

def main(arg):
    if arg.nsplit >= ep.N_SPLITS or arg.nsplit < 0:
        nth_split = ep.N_SPLITS # Using full training data
    else:
        nth_split = arg.nsplit
    pat_idx, X, X_mask, y, y_mask, othr_data = ep.load_ichi_data(data_file='ICHI_training_data.p')
    with open(ep.bam_loc + 'stats.pkl', 'rb') as fp:
        samplesStatsMap = pickle.load(fp)
    estimator = HorizontalKnn(samplesStatsMap)
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
    estimator = HorizontalKnn(samplesStatsMap) # load model
    for idx in range(estimator.model_count):
        mls = ModelLoaderSaver(os.path.join(ep.output_models, os.path.basename(__file__).rstrip('.py') + "_f" + str(idx)))
        estimator.models[idx] = mls.load_standard_model()
    pat_idx, X, X_mask, othr_data = ep.load_ichi_test_data(data_file=ep.TEST_DATA_file, is_padded=False)
    mt = ep.ModelTester(estimator, X, X_mask, normalizer=None, **othr_data)
    mt.predict()

if __name__ == "__main__":
    arg = du.parse_args()
    if arg.final:
        main_load_models(arg)
    else:
        main(arg)