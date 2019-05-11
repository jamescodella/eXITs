import numpy as np
import pandas as pd
import pickle
import os
import sys
import argparse
from scipy import interpolate

def load_base_model_predict(bam_loc, X, base_model_names, run_ids=None, normalizer=None, suffix="Final"):
    if run_ids is None:
        run_ids = [0] * len(base_model_names)
    model_data = dict()
    for clf_, run_id in zip(base_model_names, run_ids):
        with open(os.path.join(bam_loc, "{}_{}_{}".format(clf_, run_id, suffix)), "rb") as fIn:
            data = pickle.load(fIn)
            pred_y = data.pop('pred_y_test')
            if normalizer is not None:
                pred_y = normalizer(pred_y)
            model_data["model_"+ clf_] = pred_y
    return model_data

def load_base_model(bam_loc, X, n_splits, base_model_names, run_ids, get_valid=False):
    if run_ids is None:
        run_ids = [0] * len(base_model_names)
    full_data = []
    splits_used = np.arange(n_splits)
    if get_valid:
        splits_used = np.r_[splits_used, n_splits]
    for split in splits_used:
        split_data = dict()
        for clf_, run_id in zip(base_model_names, run_ids):
            # loading pred data for each base model
            with open(os.path.join(bam_loc, "{}_{}_{}".format(clf_, split, run_id)), "rb") as fIn:
                data = pickle.load(fIn)
                pred_y = data.pop('pred_y_test')
                split_data["model_"+ clf_] = pred_y
        else:
            split_data["idx"] = data["test_idx"]
        full_data.append(split_data)
    _keys = list(split_data.keys())
    # transfroming list of dict to dict of list
    modded_data = {k: list() for k in _keys}
    for split_data in full_data:
        for key in _keys:
            modded_data[key].append(split_data[key])
    idx = modded_data.pop("idx")
    all_idx = np.hstack(idx)
    for key in modded_data.keys():
        tmp = [None] * len(X)
        i = 0
        for sp in modded_data[key]:
            for elem in sp:
                tmp[all_idx[i]] = elem
                i = i + 1
        modded_data[key] = tmp
    return modded_data

def quartile_scale(dat, q=None, inverse=False):     #  Scale params by (3rd quartile - 1st quartile) as described in Luo et al. 2016
    # Input:  - a = list of numpy arrays, q = list of quartiles [1st, 2nd, 3rd], inverse = flag to invert scaling (restore to unscaled values)
    # Output: scaled list of arrays
    if q is None:
        q = [[98, 105], [3.7, 4.4], [22, 27.2], [135, 140], [25.8,34.5], [8.4, 11.3], [86.2, 94.3], [124, 274], [6.1, 12.2], [14,17.1], [12, 34], [0.69, 1.56], [98, 149]]
    if inverse is False:
        for i in range(0,dat[0].shape[0]):
            for p in range(len(dat)):
                dat[p][i][:] = (dat[p][i][:] /(q[i][1] - q[i][0]))
    else:
        for i in range(0,dat[0].shape[0]):
            for p in range(len(dat)):
                dat[p][i][:] = (dat[p][i][:] * (q[i][1] - q[i][0]))
    return dat

def calculate_nRMSE(y, yhat, y_mask, x_mask):     # Function to calculate normalized RMSE over missing values in X, but present in Y.
    # Input:  - y = actual,  yhat = prediction, y_mask = matrix same shape as y, 0 - data is missing, 1 = data is NOT missing, x_mask = matrix same shape as y, 0 - data is missing, 1 = data is NOT missing
    denom = np.zeros(y[0].shape[0])
    numer = np.zeros(y[0].shape[0])
    try:
        for p in range(0,len(y)):  # for each patient
            if yhat[p] is None:
                continue
            I = (y_mask[p] - x_mask[p])
            diff = np.abs(np.nan_to_num(y[p]) - np.nan_to_num(yhat[p])) # calculate absolute difference
            max_ypa = np.nanmax(y[p], axis=1) #  max value for each feature for this patient
            min_ypa = np.nanmin(y[p], axis=1) #  min value for each feature for this patient
            diff_denom = max_ypa - min_ypa # normalizing denominator
            diff_denom = np.repeat(diff_denom[:,np.newaxis], y[p].shape[1], axis=1) # create matrix for dividing
            div_diff = np.divide(diff,diff_denom) # abs(y-yhat)/(max_ypa - min_ypa)
            div_diff = np.square(div_diff) # square normalized absolute difference
            I_mult_diff = np.multiply(div_diff,I) # multiple by  matrix indicating missing (imputed) values
            sum_I_i_mult_diff = np.sum(I_mult_diff,axis=1)  # sum over time points
            denom_sum_I_i = np.sum(I,axis=1) # sum I over time points
            denom = np.add(denom,denom_sum_I_i)
            numer = np.add(numer, sum_I_i_mult_diff)
    except Exception as e:
        raise Exception(e)
    nRMSE = np.sqrt(np.divide(numer, denom).astype(np.float64))
    return nRMSE

def output_to_csv(prediction, X, X_mask, X_time, ID_list,loc): # Input: list of  arrays, list of  arrays, list of  arrays, list of  arrays, list of int, string
    final_prediction = []
    # Create list of predictions
    for p in range(0,len(prediction)):
        X_temp = X[p]
        X_temp[np.isnan(X_temp)] = 0
        pred_temp = prediction[p] * (1 - X_mask[p])
        pred_temp = pred_temp + X_temp
        final_prediction.append(pred_temp)
    # Create CSV files of predictions
    for p in range(0,len(final_prediction)):
        patient_ID = str(ID_list[p])
        time = X_time[p].astype(int)
        pred = np.append(final_prediction[p],time,axis=0)
        pred = np.roll(pred.T,1,axis=0)
        df = pd.DataFrame(data=pred, columns=['PCL', 'PK', 'PLCO2', 'PNA', 'HCT', 'HGB', 'MCV', 'PLT', 'WBC', 'RDW', 'PBUN', 'PCRE', 'PGLU','CHARTTIME'])
        df[[df.columns[-1]] + list(df.columns[:-1])].sort_values('CHARTTIME').to_csv(loc+patient_ID+'.csv', index=False)

def interpolate_linear_x(X, X_time, samplesStatsMap):
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
                val = max(val, samplesStatsMap[iLabTest]['min'])
                val = min(val, samplesStatsMap[iLabTest]['max'])
                piX[iLabTest, jTime] = val
        X_int.append(piX)
        sys.stdout.write('\rInterpolating: ' + ("{:.1f}".format((subject/len(X))*100)) + '%')
        sys.stdout.flush()
    sys.stdout.write('\nInterpolation done.\n')
    return X_int

def get_samples_statistics(X, labTestNames):
    samplesStatsMap = {}
    for i in range(len(labTestNames)):
        labSamples = []
        for p in range(len(X)):
            labSamples.append(X[p][i, :])
        labSamples = np.hstack(labSamples)
        labSamples = labSamples[~np.isnan(labSamples)]
        m = {}
        m['min'] = min(labSamples)
        m['max'] = max(labSamples)
        m['median'] = np.median(labSamples)
        m['mean'] = np.mean(labSamples)
        m['std'] = np.std(labSamples)
        percentiles = np.zeros(101)
        for q in range(101):
            percentiles[q] = np.percentile(labSamples, q)
        m['percentiles'] = percentiles
        samplesStatsMap[labTestNames[i]] = m
        samplesStatsMap[i] = m
    return samplesStatsMap

def parse_args(): # Global parse_args for each model file. Reads the command line options and parses the appropriate commands
    ap = argparse.ArgumentParser('program')
    ap.add_argument("-n", "--nsplit", metavar='NSPLIT', required=False, type=int, help="split value to use.") # changed to false require for compatability
    ap.add_argument("--conf", metavar='CONFIG', required=False, type=str, default=None,help="CONFIG FILE")
    ap.add_argument('-v', '--verbose', required = False, action="store_true", help="Log option: Write debug messages.") # BRITS
    ap.add_argument('-e', '--epoch', default=50, required=False, type=int, help="Model BRITS epochs.") # BRITS
    ap.add_argument('-rid', '--run_id', default=1, required=False, type=int, help="Setting run ID.") # BRITS
    ap.add_argument('-t', '--train', action="store_true", help="Log option: Write debug messages.")
    ap.add_argument('-f', '--final', default = False, action="store_true",help="Load the models and apply them on the test data to get the predictions.")
    ap.add_argument("-em", "--ensemble", required=False, type=str, choices=["Bagging-Pasting", "Bagging-Replacement", "Bagging-RS", "Bagging-RP", "XGB", "RF", "BMA", "Gate", "FFN"], help="ensemble name") # modified false for compatability
    ap.add_argument("-m", "--model", metavar='MODEL', required=False,type=str, help="Specify model for eval.") # changed to false require for compatability
    ap.add_argument("-r", "--run", metavar='RUN', required=False, type=int, default=0,help="optional param.")
    ap.add_argument("-i", "--inp",required = False, type=str, default='' ,help="Path of input data files for process_data.py. E.g. path/to/input/")
    ap.add_argument("-o", "--out",required = False, type=str, default=def_out_path ,help="Path of output data files for process_data.py. E.g. path/to/output/")
    arg = ap.parse_args()
    return arg