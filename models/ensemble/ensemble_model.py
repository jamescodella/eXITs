from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import os
import copy
import numpy as np
import logging
import yaml
from yaml import Loader, Dumper
from .util import ModelLoaderSaver, Reader, set_logger, convert, inverse_convert, correct_names
from .gate_ensemble import GateEnsemble, FFN
from .bma import BMA
from .dataloader import DataLoader, StaticDataLoader
import torch
import torch.nn as nn
import torch.optim as op
import torch.autograd as py

class StandardEnsembel():
    def __init__(self, ensemble_name, reader, mls):
        super(StandardEnsembel, self).__init__()
        self.__name__ = ensemble_name
        self.ensemble_name = ensemble_name # the name of the ensemble model has to be one of the list in the config file
        self.reader = reader # reader for the config file
        self.mls = mls # model loader saver object
        self._construct_regression_models(ensemble_name)

    def _construct_regression_models(self, estimator_name):
        logging.info("_construct_regression_models for {} ...".format(estimator_name))
        self.self_param_spase = True
        if estimator_name == "Bagging-Pasting":
            self.estimator = BaggingRegressor(DecisionTreeRegressor(),bootstrap=False, max_features=1.0)
        elif estimator_name == "Bagging-Replacement":
            self.estimator = BaggingRegressor(DecisionTreeRegressor(),bootstrap=True, max_features=1.0)
        elif estimator_name == "Bagging-RS":
            self.estimator = BaggingRegressor(DecisionTreeRegressor())
        elif estimator_name == "Bagging-RP":
            self.estimator = BaggingRegressor(DecisionTreeRegressor())
        elif estimator_name == "XGB":
            self.estimator = XGBRegressor()
        elif estimator_name == "RF":
            self.estimator = RandomForestRegressor()
        elif estimator_name == "BMA":
            self.estimator = BMA()
        elif estimator_name == "Gate":
            self.self_param_spase = False
            self.estimator = GateEnsemble(self.reader.get_gate_ensemble().get_structure())
            self.criterion = nn.MSELoss()
            self.optimizer = op.SGD(params=self.estimator.parameters(), **self.reader.get_gate_ensemble().get_optimizer_params(), weight_decay=.2)
        elif estimator_name == "FFN":
            self.self_param_spase = False
            self.estimator = FFN(self.reader.get_fnn_ensemble().get_structure())
            self.criterion = nn.MSELoss()
            self.optimizer = op.SGD(params=self.estimator.parameters(), **self.reader.get_fnn_ensemble().get_optimizer_params(), weight_decay=.2)
        else :
            raise Exception

        self.estimators = dict()
        if self.self_param_spase :
            self.estimators[self.ensemble_name] = self._define_param_space (self.estimator, estimator_name)
        else :
            self.estimators[self.ensemble_name] = self.estimator

    def _define_param_space (self, mdl, estimator_name):
        if estimator_name == "Bagging-Pasting" : # subsets of samples
            param_space = {'n_estimators':[10,50,100], 'max_samples':[.6,.8, 1.0]}
            reg_param_search = GridSearchCV(mdl, param_grid=param_space, cv=3)
        elif estimator_name == "Bagging-Replacement" : # with replacement
            param_space = {'n_estimators':[10,50,100], 'max_samples':[.6,.8, 1.0]}
            reg_param_search = GridSearchCV(mdl, param_grid=param_space, cv=3)
        elif estimator_name == "Bagging-RS" : # random-subspace
            param_space = {'n_estimators':[10,50,100], 'max_features':[.3,.5,.7,1.0]}
            reg_param_search = GridSearchCV(mdl, param_grid=param_space, cv=3)
        elif estimator_name == "Bagging-RP" : # random-patches
            param_space = {'n_estimators':[10,50,100], 'max_samples':[.6,.8, 1.0], 'max_features':[.3,.5,.7,1.0]}
            reg_param_search = GridSearchCV(mdl, param_grid=param_space, cv=3)
        elif estimator_name is "XGB":
            param_space = {'max_depth': [2,3,4], 'n_estimators':[10,50,100]}
            reg_param_search = GridSearchCV(mdl, param_grid=param_space, cv=3)
        elif estimator_name is "RF":
            param_space = {'max_depth': [3, 5, 7], 'n_estimators':[10,50,100]}
            reg_param_search = GridSearchCV(mdl, param_grid=param_space, cv=3)
        else:
            reg_param_search = mdl
        return reg_param_search

    def _train(self, train_data_loader, val_data_loader, mls, name, **kwargs): #Train the model on training data for multiple epochs and choosing the one that has the best metric on the validation data
        torch.manual_seed(10)
        self.estimators[name].weights_init()
        # Training the model
        best_score = np.inf # np.inf or 0 ... score on the validation set
        num_epochs = kwargs['num_epochs'] if 'num_epochs' in kwargs else 1000
        for epoch in range(num_epochs):
            train_data_iterator = train_data_loader.data_iterator() # get the iterator for batchifying the training data
            ypred_model, loss_model = self.estimators[name].fit(train_data_iterator, self.optimizer, self.criterion)   # train the model
            logging.debug("Validation ...")
            score = self.estimators[name].score(val_data_loader) # dictionary of metrics
            if best_score > score: # this could be less (accuracy) of greate (mse)
                best_score = score
                self.mls.save_model(self.estimators[name].state_dict(), self.optimizer.state_dict(), best=True)
                logging.debug("- best metrics on validation set: " + str(score))
            if epoch%10 == 0 :
                logging.debug("Epoch %d: loss is %.3f" % (epoch, loss_model.item()))
            if epoch%100 == 0 : # save the model
                self.mls.save_model(self.estimators[name].state_dict(), self.optimizer.state_dict())

    def _get_feature_prediction(self, othr_data, base_names, f_idx):
        X_models = convert(othr_data[base_names[0]])[:,f_idx] # np.empty(shape=(0,y_train.shape[0]*len(base_names)))
        for m in base_names[1:]:
            X_models = np.vstack((X_models, convert(othr_data[m])[:,f_idx]))
        return X_models.transpose()

    def _remove_nans(self, X, y):
        mock = ~np.isnan(y)
        y = y[mock]
        X = X[mock,:]
        return X, y

    def _remove_nans_v2(self, X, y, X2):
        mock =  (~np.isnan(y)) & (~np.isnan(X2).any(axis=1))
        y = y[mock]
        X = X[mock,:]
        X2 = X2[mock,:]
        return X, y, X2

    def fit (self, X_train_lst, y_train_lst, X_mask_train_lst, y_mask_train_lst, valid_data=None, **othr_data):
        X_train = convert (X_train_lst)
        y_train = convert (y_train_lst)
        base_names = correct_names(self.reader.get_base_models())
        if self.self_param_spase:
            for i in range(y_train.shape[1]):
                name = self.ensemble_name + "_f" + str(i)
                if name not in self.estimators.keys():
                    self.estimators[name] = copy.deepcopy(self.estimators[self.ensemble_name])
                X_models = self._get_feature_prediction(othr_data, base_names, i)
                X, y = self._remove_nans(X_models,y_train[:,i])
                self.estimators[name].fit(X, y)
        elif self.ensemble_name =='Gate' or self.ensemble_name=='FFN':
            if self.ensemble_name =='Gate' :
                ensemble_block_def = self.reader.get_gate_ensemble()
            elif self.ensemble_name=='FFN':
                ensemble_block_def = self.reader.get_fnn_ensemble()

            kwargs = {'batch_size':ensemble_block_def.get_batch_size(), 'cuda':ensemble_block_def.get_cuda()}
            for i in range(y_train.shape[1]):
                name = self.ensemble_name + "_f" + str(i)
                if name not in self.estimators.keys():
                    self.estimators[name] = copy.deepcopy(self.estimators[self.ensemble_name])
                X_models = self._get_feature_prediction(othr_data, base_names, i)
                X_m, y, X = self._remove_nans_v2(X_models,y_train[:,i], X_train)
                c = int(X_m.shape[0]*.9)
                X_m_val, y_val, X_val = X_m[c+1:,:], y[c+1:,], X[c+1:,:]
                X_m, y, X = X_m[:c,:], y[:c,], X[:c,:]
                add_dct = {'batch_size':ensemble_block_def.get_batch_size(), 'cuda':ensemble_block_def.get_cuda()}
                train_data_loader = StaticDataLoader(X, y, X_m, add_dct)
                val_data_loader = StaticDataLoader(X_val, y_val, X_m_val, add_dct)
                self._train(train_data_loader, val_data_loader, self.mls, name, num_epochs=ensemble_block_def.get_num_epochs())

    def predict(self, X_test_lst, X_mask_test_lst, **othr_data): # Compute predictions
        num_features = len(self.estimators.keys())-1
        # use the filling data in order to not get error when performing forward function on Nan values
        X_test_lst_filled = othr_data['X_filled']
        seq_lengths = [X_test_lst_filled[j].shape[1] for j in range(0,len(X_test_lst_filled))]
        X_test = convert (X_test_lst_filled)
        y_predict = np.empty(shape=(X_test.shape[0],num_features))
        base_names = correct_names(self.reader.get_base_models())

        for i in range(num_features):
            X_models = self._get_feature_prediction(othr_data, base_names, i) # get the prediction from individual models for the feature i
            name = self.ensemble_name + "_f" + str(i)
            if self.self_param_spase:
                y_predict[:,i] = self.estimators[name].predict(X_models)
            else :
                if self.ensemble_name =='Gate' :
                    ensemble_block_def = self.reader.get_gate_ensemble()
                elif self.ensemble_name=='FFN':
                    ensemble_block_def = self.reader.get_fnn_ensemble()
                add_dct = {'batch_size':ensemble_block_def.get_batch_size(), 'cuda':ensemble_block_def.get_cuda()}
                test_data_loader = StaticDataLoader(X_test, None, X_models, add_dct)
                y_predict[:,i] = self.estimators[name].predict(test_data_loader).detach().numpy()
        return inverse_convert(seq_lengths, y_predict)