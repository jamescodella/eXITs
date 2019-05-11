import estimator_pipeline as ep
from ensemble.util import ModelLoaderSaver
import data_utils as du
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Masking
import pickle
import warnings
import argparse
import os
from datetime import datetime
warnings.filterwarnings('ignore')

class BiLSTM(object):
    def __init__(self, in_dim=(None, 13), out_dim=(13), num_hidden=100, r_dropout=0.3, padding_value=-1):
        super(BiLSTM, self).__init__()
        self.model = self.create_model(in_dim=in_dim, out_dim=out_dim, num_hidden=num_hidden, r_dropout=r_dropout, padding_value = padding_value)
        self.seq_lengths = []
        self.__name__ = "BiLSTM"
        
    def create_model(self, in_dim, out_dim, num_hidden, r_dropout, padding_value):
        model_BiLSTM = Sequential()
        model_BiLSTM.add(Masking(mask_value=padding_value, input_shape=in_dim))
        model_BiLSTM.add(Bidirectional(LSTM(num_hidden, input_shape=in_dim, return_sequences=True, recurrent_dropout=r_dropout)))
        model_BiLSTM.add(Dense(out_dim))
        model_BiLSTM.compile(loss='mean_squared_error', optimizer='adam')
        return model_BiLSTM
    
    def prep_data(self, X, y, X_mask, y_mask, valid_data, **kwargs):
        # prep X data        
        X_, m_, = kwargs['X_filled'], X_mask
        self.seq_lengths = [((kwargs['X_filled'][j] > 0).sum(axis=0) > 0).sum() for j in range(0,len(X))]
        X_, m_  = np.asarray(X_), np.asarray(m_)
        X_,  m_  = X_.swapaxes(1,2), m_.swapaxes(1,2) 
        X = X_ 
        if valid_data is not None:
            X_valid_, m_valid_, y_valid_ = (valid_data['othr_data']['X_filled'], valid_data['X_mask'],  valid_data['othr_data']['y_filled'])
            X_valid_, m_valid_, y_valid_ = np.asarray(X_valid_), np.asarray(m_valid_),  np.asarray(y_valid_)
            X_valid_, m_valid_, y_valid_ = X_valid_.swapaxes(1,2), m_valid_.swapaxes(1,2), y_valid_.swapaxes(1,2)
            X_valid = X_valid_ # np.concatenate((X_valid_, m_valid_, d_valid_), axis=2)
        if y is not None:
            y_ = np.asarray(kwargs['y_filled'])
            y_ = y_.swapaxes(1,2)
        if y is not None:
            return X, y_, X_valid, y_valid_
        else:
            return X
    
    def fit(self, X, y, X_mask, y_mask, valid_data, normalizer=None, num_epochs=500, batch_size=64, verbose=1, **kwargs):
        X_train, y_train, X_valid, y_valid = self.prep_data(X, y, X_mask, y_mask, valid_data, **kwargs)
        es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=10,min_delta=0.0001)
        tb_name = self.__name__ + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tb = keras.callbacks.TensorBoard(log_dir='bam/loss_logs/'+tb_name, histogram_freq=0, write_graph=True, write_images=True)
        self.train_hist = self.model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=num_epochs, batch_size=batch_size, verbose=verbose,callbacks=[es, tb])

    def predict(self, X, X_mask, **kwargs):
        X = self.prep_data(X = X, y = None, X_mask = X_mask, y_mask = None, valid_data = None, **kwargs)
        model_prediction = self.model.predict(X)
        pred_list=[]
        for i in range(0,model_prediction.shape[0]):
            pred_list.append(model_prediction[i][0:self.seq_lengths[i],:].T)
        return pred_list

    def save(self):
        pass

def main(arg):
    if arg.nsplit >= ep.N_SPLITS or arg.nsplit < 0:
        nth_split = ep.N_SPLITS # Using full training data
    else:
        nth_split = arg.nsplit
    pat_idx, X, X_mask, y, y_mask, othr_data = ep.load_ichi_data(data_file=ep.DATA_file)
    estimator = BiLSTM()
    mt = ep.ModelTrainer(estimator, X, y, X_mask, y_mask, normalizer=du.quartile_scale, **othr_data)
    mt.fit_predict(nth_split=nth_split,run_id=0)     # nth_split is a value between 0 and 4
    if nth_split == ep.N_SPLITS: # Also save the model
        mls = ModelLoaderSaver(os.path.join(ep.output_models, os.path.basename(__file__).rstrip('.py')))
        mls.save_keras_model(mt.estimator_.model)

def main_load_models(arg):    # load model
    estimator = BiLSTM()
    mls = ModelLoaderSaver(os.path.join(ep.output_models, os.path.basename(__file__).rstrip('.py')))
    estimator.model = mls.load_keras_model()
    pat_idx, X, X_mask, othr_data = ep.load_ichi_test_data(data_file=ep.TEST_DATA_file) # apply the model to get the prediciton
    mt = ep.ModelTester(estimator, X, X_mask, normalizer=du.quartile_scale, **othr_data)
    mt.predict()

if __name__ == "__main__":
    arg = du.parse_args()
    if arg.final:
        main_load_models(arg)
    else:
        main(arg)