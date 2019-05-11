import torch
import torch.optim as op
import torch.autograd as py
from sklearn.metrics import mean_squared_error
from math import sqrt
import logging
import yaml
from yaml import Loader, Dumper
import numpy as np
import pickle
from tensorflow.keras.models import load_model

class ModelLoaderSaver():
    # Loads and saves model into json file
    def __init__(self, file_path):
        self.file_path = file_path

    def save_model(self, model_state, opt_state, best=False):
        state = {'model_state': model_state,'opt_state': opt_state}
        if best:
            torch.save(state, self.file_path + '_best.pth')
        else:
            torch.save(state, self.file_path)

    def load_model(self, model, optimizer=None, best=True): # Loads the model from self.file_path file. If best=True it loads the model that achieves the best metric on the validation set
        # Parameters:  best: if true loads the best model on validation set
        if best:
            state = torch.load(self.file_path + '_best.pth')
        else:
            state = torch.load(self.file_path)
        model.load_state_dict(state['model_state'])
        if optimizer is not None: optimizer.load_state_dict(state['opt_state'])

    def save_standard_model(self, model):
        with open(self.file_path,'wb') as fout:
            pickle.dump(model, fout)

    def save_keras_model(self, model):
        model.save(self.file_path)

    def load_standard_model(self):
        with open(self.file_path,'rb') as fin:
            return pickle.load(fin)

    def load_keras_model(self):
        return load_model(self.file_path)

class Reader():
    # Reads the lis of ensemble models from the yaml file
    def __init__(self, file_path):
        super(Reader, self).__init__()
        with open(file_path) as f:
            self.params = yaml.load(f, Loader=Loader)
            logging.debug(self.params)

    def get_standard_ensembles(self):
        return self.params['ensemble_models']['standard_ensemble']

    def get_gate_ensemble(self):
        return NNReader(self.params['ensemble_models']['gate_ensemble'])

    def get_fnn_ensemble(self):
        return NNReader(self.params['ensemble_models']['feedforward_ensemble'])

    def get_base_models(self):
        return self.params['base_models']

class NNReader():
    def __init__(self, params):
        super(NNReader, self).__init__()
        self.params = params

    def get_structure(self):
        return self.params['structure']

    def get_batch_size(self):
        return self.params['batch_size']

    def get_num_epochs(self):
        return self.params['num_epochs']

    def get_optimizer_params(self):
        return self.params['optimizer_params']

    def get_cuda(self):
        return self.params['cuda']


def set_logger(log_path, level):
    # Set the logger to log info in terminal and file `log_path`
    # Parameters: log_path: (string) where to log
    logger = logging.getLogger()
    logger.setLevel(level)
    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)
        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

def metric(y_pred, y):
    return sqrt(mean_squared_error(y, y_pred))

def convert (lst): # lst is a list. lst[i] is 2 dim numpy array FxT. This converts lst to be 2 dim array (P*T)xF
    return np.hstack(lst).transpose()

def inverse_convert(seq_lengths, y):
    preds = []
    sum_p = 0
    for p in seq_lengths:
        preds.append(y[sum_p:sum_p+p,:].transpose())
        sum_p = sum_p+p
    return preds

def correct_names(lst):
    return ['model_' + i for i in lst]