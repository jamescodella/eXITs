import pandas as pd
import numpy as np
import pickle
import os
import copy
from sklearn.model_selection import StratifiedKFold, train_test_split
import data_utils as du
import logging
from six import iteritems
bam_loc = 'bam/data/'
output_models = 'bam/models/'
DATA_file = 'ICHI_PADDED_training_data.p'
TEST_DATA_file = 'ICHI_PADDED_test_data.p'
CV_file = 'cv_splits.pkl'
N_SPLITS = 5

def load_ichi_data(data_file, is_padded = True):
    if not is_padded:
        data_file = data_file.replace('_PADDED', '')
    with open(os.path.join(bam_loc, data_file), 'rb') as fIn:
        # loading dictionary containing [X, X_mask, y, y_mask, X_filled, X_diff, X_time]
        data = pickle.load(fIn)
    X = data.pop('X')     # each element, is a list of N patients, each elements FxT np array
    X_mask = data.pop("X_mask")
    y = data.pop("y")
    y_mask = data.pop("y_mask")
    pat_idx = np.atleast_1d(np.arange(len(X)))
    return pat_idx, X, X_mask, y, y_mask, data

def load_ichi_test_data(data_file, is_padded = True):
    if not is_padded:
        data_file = data_file.replace('_PADDED', '')
    with open(os.path.join(bam_loc, data_file), 'rb') as fIn:
        # loading dictionary containing [X, X_mask, y, y_mask, X_filled, X_diff, X_time]
        data = pickle.load(fIn)

    X = data.pop('X')     # each element, is a list of N patients, each elements FxT np array
    X_mask = data.pop("X_mask")
    pat_idx = np.atleast_1d(np.arange(len(X)))
    return pat_idx, X, X_mask, data

def get_stratifier(mask):     #compute stratifier from mask.
    # Steps:  1. compute present values for each elements of mask; 2. calculate quantile;  3. bin missing values into quantiles
    # Returns:  * stratifier value; * missing counts
    presence = np.zeros(len(mask))
    for idx, p in enumerate(mask):
        tmp = pd.DataFrame(p.T).replace(-1, np.NAN).dropna()  # extracting unpadded, T X F
        presence[idx] = (tmp == 1).sum(axis=1).mean()  # average number of missing per time point

    bins = np.array([np.min(presence), np.quantile(presence, 0.33), np.quantile(presence, 0.66)])
    # s3
    stratif_ = np.digitize(presence, bins)
    return stratif_, presence

class BaseEstimator(object):
    def fit(self, X, y, X_mask, y_mask, valid_data=None, **kwargs):
        return

    def predict(self, X, x_mask, **kwargs):
        y_pred = None
        return y_pred

class ModelTester(object):
    def __init__(self, estimator, X, X_mask, normalizer=None, **othr_data):
        super(ModelTester, self).__init__()
        self.X_mask = X_mask
        self.normalizer = normalizer
        if self.normalizer is not None:
            self.X = self.normalizer(X)
            self.othr_data = {k: self.normalizer(v) if k.endswith('_filled') or k.startswith('model_') else v for k, v in iteritems(othr_data)}
        else:
            self.X = X
            self.othr_data = othr_data
        self.estimator_ = estimator

    def predict(self, run_id=0):
        pred_y_test = self.estimator_.predict(self.X, self.X_mask, **self.othr_data)
        return self._save_predict(pred_y_test)
        
    def _save_predict(self, pred_y_test, run_id=0):
        if self.normalizer is not None:
            pred_y_test = self.normalizer(pred_y_test, inverse=True)

        with open(os.path.join(bam_loc, "{}_{}_Final".format(self.estimator_.__name__, run_id)),
                  'wb') as fOut:
            data = dict(pred_y_test=pred_y_test, X=self.X, X_mask=self.X_mask, othr_data=self.othr_data, run_id=run_id)
            pickle.dump(data, fOut)
        return pred_y_test

class StackedTester(ModelTester):
    def __init__(self, ensemble, X, X_mask, base_models, base_run_ids=None, normalizer=None, use_valid=True, **othr_data):
        super(StackedTester, self).__init__(ensemble, X, X_mask, normalizer=None, **othr_data)
        base_model_data = du.load_base_model_predict(bam_loc, self.X, base_models, base_run_ids,normalizer=normalizer)
        self.othr_data.update(base_model_data)

class ModelTrainer(object):
    def __init__(self, estimator, X, y, X_mask, y_mask, normalizer=None, **othr_data):
        super(ModelTrainer, self).__init__()
        with open(os.path.join(bam_loc, CV_file), 'rb') as fIn:
            self.cv_indices = pickle.load(fIn)
        self.X = X
        self.y = y
        self.X_mask = X_mask
        self.y_mask = y_mask
        self.normalizer = normalizer
        self.othr_data = othr_data
        self.n_splits = len(self.cv_indices["cv"])
        self.estimator_ = estimator

    def _list_slice(self, list_, slicer_, normed=False):
        ret = [list_[idx] for idx in slicer_]
        if (self.normalizer is not None) and normed:
            ret = self.normalizer(copy.deepcopy(ret))
        return ret

    def get_indexed_data(self, idx, **arrays):
        X = self._list_slice(self.X, idx, True)
        y = self._list_slice(self.y, idx, True)
        X_mask = self._list_slice(self.X_mask, idx)
        y_mask = self._list_slice(self.y_mask, idx)
        array_data = dict()
        if bool(arrays):
            for key in arrays:
                if key.endswith('_filled') or key.startswith('model_'):
                    _norm = True
                else:
                    _norm = False
                array_data[key] = self._list_slice(arrays[key], idx, _norm)

        othr_data = None
        if self.othr_data is not None:
            othr_data = dict()
            for key in self.othr_data:
                if key.endswith('_filled') or key.startswith('model_'):
                    _norm = True
                else:
                    _norm = False
                othr_data[key] = self._list_slice(self.othr_data[key], idx, _norm)
        othr_data.update(array_data)
        return dict(idx=idx, X=X, y=y, X_mask=X_mask, y_mask=y_mask, othr_data=othr_data)

    def get_valid(self, **arrays):
        valid_idx = self.cv_indices["pat_valid"]
        ret = self.get_indexed_data(valid_idx, **arrays)

        return ret

    def get_split(self, nth_split, **arrays):
        # returns a tuple of (train, test) train contains (train_idx, X_train, y_train, X_mask_train, y_mask_train, othr_train)
        if nth_split < self.n_splits and nth_split >= 0:
            train_idx, test_idx = self.cv_indices["cv"][nth_split]
            # Required data
            train_ = self.get_indexed_data(train_idx, **arrays)
            X_train = train_["X"]
            y_train = train_["y"]
            X_mask_train = train_["X_mask"]
            y_mask_train = train_["y_mask"]
            othr_train = train_["othr_data"]
            test_ = self.get_indexed_data(test_idx, **arrays)
            X_test = test_["X"]
            y_test = test_["y"]
            X_mask_test = test_["X_mask"]
            y_mask_test = test_["y_mask"]
            othr_test = test_["othr_data"]

        elif (nth_split < 0) or (nth_split == self.n_splits): # self.nth_split
            # For last fold, return all
            train_idx = self.cv_indices['pat_used']
            train_ = self.get_indexed_data(train_idx, **arrays)
            X_train = train_["X"]
            y_train = train_["y"]
            X_mask_train = train_["X_mask"]
            y_mask_train = train_["y_mask"]
            othr_train = train_["othr_data"]

            test_idx = self.cv_indices['pat_valid']
            test_ = self.get_indexed_data(test_idx, **arrays)
            X_test = test_["X"]
            y_test = test_["y"]
            X_mask_test = test_["X_mask"]
            y_mask_test = test_["y_mask"]
            othr_test = test_["othr_data"]
        ret = ((train_idx, X_train, y_train, X_mask_train, y_mask_train, othr_train),
               (test_idx, X_test, y_test, X_mask_test, y_mask_test, othr_test))
        return ret

    def fit_predict(self, nth_split, use_valid=True, run_id=0):
        estimator_ = self.estimator_
        ((train_idx, X_train, y_train, X_mask_train, y_mask_train, othr_train),
         (test_idx, X_test, y_test, X_mask_test, y_mask_test, othr_test)) = self.get_split(nth_split)
        if use_valid:
            valid_data = self.get_valid()
        else:
            valid_data = None

        estimator_.fit(X_train, y_train, X_mask_train, y_mask_train, valid_data=valid_data, **othr_train)
        pred_y_test = None
        if nth_split < self.n_splits:
            fName = os.path.join(bam_loc, "{}_{}_{}".format(estimator_.__name__, nth_split, run_id))
        elif nth_split < 0 or nth_split == self.n_splits:
            fName = os.path.join(bam_loc, "{}_{}_{}".format(self.estimator_.__name__, self.n_splits, run_id))
        pred_y_test = estimator_.predict(X_test, X_mask_test, **othr_test)
        if self.normalizer is not None:
            pred_y_test = self.normalizer(pred_y_test, inverse=True)
            y_test = self.normalizer(y_test, inverse=True)
        with open(fName, 'wb') as fOut:
            data = dict(test_idx=test_idx, pred_y_test=pred_y_test, X_test=X_test, y_test=y_test, X_mask_test=X_mask_test, y_mask_test=y_mask_test, othr_test=othr_test, run_id=run_id)
            pickle.dump(data, fOut)
        return pred_y_test

class StackedEnsemble(ModelTrainer):
    def __init__(self, ensemble, X, y, X_mask, y_mask, base_models, base_run_ids=None,
                 normalizer=None, use_valid=True, **othr_data):
        super(StackedEnsemble, self).__init__(ensemble, X, y, X_mask, y_mask, normalizer=None, **othr_data)
        self.nth_split = self.n_splits  # last split
        self.base_model_data = self._load_base_preds(base_models, base_run_ids, use_valid)
        self.normalizer = normalizer

    def _load_base_preds(self, base_model_names, run_ids, use_valid):
        logging.info("_load_base_preds function ...")
        modded_data = du.load_base_model(bam_loc, self.X, self.n_splits, base_model_names, run_ids, get_valid=use_valid)
        return modded_data

    def fit_predict(self, nth_split, valid=True, run_id=0):
        base_model_data = self.base_model_data

        ((train_idx, X_train, y_train, X_mask_train, y_mask_train, othr_train),
         (test_idx, X_test, y_test, X_mask_test, y_mask_test, othr_test)) = self.get_split(nth_split, **base_model_data)
        if valid:
            valid_data = self.get_valid(**base_model_data)
        else:
            valid_data = None
        self.estimator_.fit(X_train, y_train, X_mask_train, y_mask_train, valid_data=valid_data, **othr_train)
        pred_y_test = None
        if nth_split < self.n_splits:
            fName = os.path.join(bam_loc, "{}_{}_{}".format(self.estimator_.__name__, nth_split, run_id))
        elif nth_split < 0 or nth_split == self.n_splits:
            fName = os.path.join(bam_loc, "{}_{}_{}".format(self.estimator_.__name__, self.n_splits, run_id))
    
        pred_y_test = self.estimator_.predict(X_test, X_mask_test, **othr_test)
        if self.normalizer is not None:
            pred_y_test = self.normalizer(pred_y_test, inverse=True)
            y_test = self.normalizer(y_test, inverse=True)
        with open(fName, 'wb') as fOut:
            data = dict(test_idx=test_idx, pred_y_test=pred_y_test, X_test=X_test, y_test=y_test, X_mask_test=X_mask_test, y_mask_test=y_mask_test, othr_test=othr_test, run_id=run_id)
            pickle.dump(data, fOut)
        return pred_y_test

    def gen_hold_out(self, valid=True, run_id=0): # This version doesn't parallelize code. Only use this if you don't want to parallelize
        raise NotImplementedError()
        pred = self.fit_predict(self.estimator_, nth_split=self.n_splits, valid=valid, run_id=run_id)
        return pred

if __name__ == "__main__":
    # Standalone run creates the train, valid folds
    pat_idx, X, X_mask, y, y_mask, othr_data = load_ichi_data(data_file=DATA_file)
    yhat, y_presence = get_stratifier(y_mask)
    _used, _valid = train_test_split(pat_idx, test_size=267, stratify=yhat)
    pat_used = pat_idx[_used]
    pat_valid = pat_idx[_valid]
    yhat_used = yhat[_used]  # to be used for stratified KFold
    print("Hold out summary", len(pat_used), len(pat_valid))
    skf = StratifiedKFold(n_splits=N_SPLITS)
    cv = [(pat_used[_train], pat_used[_test])
          for _train, _test in skf.split(pat_used, yhat_used)]
    with open(os.path.join(bam_loc, CV_file), 'wb') as fOut:
        pickle.dump(dict(pat_used=pat_used, pat_valid=pat_valid, cv=cv), fOut)