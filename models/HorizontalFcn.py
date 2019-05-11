import numpy as np
import pickle
import argparse
from sklearn.model_selection import train_test_split
import keras.optimizers
import os
import estimator_pipeline as ep
from ensemble.util import ModelLoaderSaver
import data_utils as du
import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from HorizontalBase import HorizontalBase
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class HorizontalFcn(HorizontalBase):
    def __init__(self, samplesStatsMap, model_count=13):
        super(HorizontalFcn, self).__init__("HorizontalFcn", samplesStatsMap, model_count)
        self.models = []
        for i in range(model_count):
            self.models.append(self.getModel())

    def getModel(self):
        model = Sequential()
        actiFunc = 'linear'
        model.add(Dense(10, activation=actiFunc, input_dim=13))
        model.add(Dropout(0.25))
        model.add(Dense(10, activation=actiFunc))
        model.add(Dropout(0.25))
        model.add(Dense(8, activation=actiFunc))
        model.add(Dropout(0.25))
        model.add(Dense(4, activation=actiFunc))
        model.add(Dropout(0.25))
        model.add(Dense(1, activation="linear"))
        opt = keras.optimizers.Adam(lr=0.001)  # , decay=0.0008/20, clipnorm=1)
        model.compile( loss='mean_squared_error', optimizer=opt, metrics=['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'] )
        return model

    def fit(self, X, y, X_mask, y_mask, valid_data, normalizer=None, num_epochs=100, batch_size=64, verbose=1, **kwargs):
        X_int = kwargs['X_interpolated_linear']
        X = self.normalizeZScore(X)
        X_int = self.normalizeZScore(X_int)
        X_flat = np.hstack(X)
        X_int_flat = np.hstack(X_int)
        y_flat = np.hstack(y)
        tb_name = self.__name__ + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")       
        callbacks_list = [ keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min'), keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.0001, mode='min'), keras.callbacks.TensorBoard(log_dir='bam/loss_logs/'+tb_name, histogram_freq=0, write_graph=True, write_images=True) ]
        #keras.callbacks.ModelCheckpoint(modelWeightPath, monitor='val_loss', save_best_only=True, mode='min')

        print('Training')
        for iLab in range(self.model_count):
            print('\t' + str(iLab + 1) + '/' + str(self.model_count))
            X_col, y_col = self.prepareXY(X_flat, X_int_flat, y_flat, iLab, remove_y_nan=True)
            trainColX, testColX, trainColY, testColY = train_test_split(X_col, y_col, test_size=0.33, random_state=42)
            self.models[iLab].fit(trainColX, trainColY, validation_data=(testColX, testColY), callbacks=callbacks_list, batch_size=32, verbose=verbose, epochs=num_epochs)

    def predict(self, X, X_mask, **kwargs):
        X_original = X.copy()
        X = self.normalizeZScore(X.copy())
        X_int = kwargs['X_interpolated_linear']
        X_int = self.normalizeZScore(X_int)
        X_flat = np.hstack(X)
        X_int_flat = np.hstack(X_int)
        samplesPerSubject = [px.shape[1] for px in X]
        y_pred_flat_list = []
        print('Testing')
        for iLab in range(self.model_count):
            print('\t' + str(iLab+1) + '/' + str(self.model_count))
            X_col, y_col = self.prepareXY(X_flat, X_int_flat, X_int_flat, iLab, remove_y_nan=False)
            y_pred_p = self.models[iLab].predict(X_col)
            y_pred_p = [x[0] for x in y_pred_p]
            y_pred_flat_list.append(y_pred_p)
        y_pred_flat = np.vstack(y_pred_flat_list).transpose()
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

    estimator = HorizontalFcn(samplesStatsMap)
    mt = ep.ModelTrainer(estimator, X, y, X_mask, y_mask, normalizer=None, **othr_data)
    mt.fit_predict(nth_split=nth_split)     # a value between 0 and 4
    if nth_split == ep.N_SPLITS:    # Also save the model
        for idx, model in enumerate(mt.estimator_.models):
            mls = ModelLoaderSaver(os.path.join(ep.output_models, os.path.basename(__file__).rstrip('.py') + "_f" + str(idx)))
            mls.save_keras_model(model)

def main_load_models(arg):
    with open(ep.bam_loc + 'stats.pkl', 'rb') as fp:
        samplesStatsMap = pickle.load(fp)
    # load model
    estimator = HorizontalFcn(samplesStatsMap)
    for idx in range(estimator.model_count):
        mls = ModelLoaderSaver(os.path.join(ep.output_models, os.path.basename(__file__).rstrip('.py') + "_f" + str(idx)))
        estimator.models[idx] = mls.load_keras_model()
    pat_idx, X, X_mask, othr_data = ep.load_ichi_test_data(data_file=ep.TEST_DATA_file, is_padded=False)
    mt = ep.ModelTester(estimator, X, X_mask, normalizer=None, **othr_data)
    mt.predict()

if __name__ == "__main__":
    arg = du.parse_args()
    if arg.final:
        main_load_models(arg)
    else:
        main(arg)