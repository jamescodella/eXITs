from estimator_pipeline import load_ichi_data, load_ichi_test_data, StackedEnsemble, ModelTester, TEST_DATA_file, DATA_file, bam_loc, StackedTester
from data_utils import quartile_scale, parse_args
import argparse
import os
import logging
from ensemble.util import set_logger
from ensemble.util import Reader
from ensemble.util import ModelLoaderSaver
from ensemble.ensemble_model import StandardEnsembel
from six import iteritems

def main_load_models(arg):
    reader = Reader(os.path.join(dir_name , "ensemble/config.yml"))
    base_models = reader.get_base_models()
    # load models
    standard_ensemble = StandardEnsembel(arg.ensemble, reader, None)
    if arg.ensemble!='Gate' and arg.ensemble !='FFN':

        mls = ModelLoaderSaver(os.path.join(dir_name , bam_loc + arg.ensemble ))
        models = mls.load_standard_model() # dictionary of all feature models (model for each feature)
        for name, model in iteritems(models):
            standard_ensemble.estimators[name] = model
    else:
        for i in range(len(standard_ensemble.estimator_.estimators)-1):
            name = arg.ensemble + "_f" + str(i)
            mls = ModelLoaderSaver(os.path.join(dir_name , bam_loc + name ))
            mls.load_model(standard_ensemble.estimator_.estimators[name])    # apply the model to get the prediciton
    pat_idx, X, X_mask, othr_data = load_ichi_test_data(data_file=TEST_DATA_file,is_padded = False)
    mt = StackedTester(standard_ensemble, X, X_mask, base_models, normalizer=quartile_scale, **othr_data)
    mt.predict()

def main(arg):
    reader = Reader(os.path.join(dir_name , "ensemble/config.yml"))
    base_models = reader.get_base_models()
    mls = ModelLoaderSaver(os.path.join(dir_name , bam_loc + arg.ensemble + "_" + str(arg.nsplit)))
    pat_idx, X, X_mask, y, y_mask, othr_data = load_ichi_data(data_file=DATA_file, is_padded=False)
    stacked_ensemble = StackedEnsemble(StandardEnsembel(arg.ensemble, reader, mls), X, y, X_mask, y_mask, base_models, normalizer=quartile_scale, **othr_data)
    if arg.train:
        nth_split = arg.nsplit
        stacked_ensemble.fit_predict(nth_split=nth_split)     # a value between 0 and 4
    else:
        stacked_ensemble.fit_predict(nth_split=5)     # a value between 0 and 4
        if arg.ensemble=='Gate' or arg.ensemble=='FFN':
            for i in range(len(stacked_ensemble.estimator_.estimators)-1):
                name = arg.ensemble + "_f" + str(i)
                mls = ModelLoaderSaver(os.path.join(dir_name , bam_loc + name ))
                mls.save_model(stacked_ensemble.estimator_.estimators[name].state_dict(), None)
        else:
            model = dict()
            for i in range(len(stacked_ensemble.estimator_.estimators)-1):
                name = arg.ensemble + "_f" + str(i)
                model[name] = stacked_ensemble.estimator_.estimators[name]
            mls = ModelLoaderSaver(os.path.join(dir_name , bam_loc + arg.ensemble ))
            mls.save_standard_model(model)
    return

def warn(*args, **kwargs):
    pass

if __name__ == "__main__":
    import warnings
    warnings.warn = warn
    dir_name = os.path.dirname(os.path.realpath(__file__))
    set_logger(os.path.join(dir_name , "log/log.log"), logging.DEBUG)
    logging.info("Starting ensemble model")
    arg = parse_args()
    if arg.final:
        main_load_models(arg)
    else:
        main(arg)