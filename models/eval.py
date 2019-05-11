import pandas as pd
import numpy as np
import data_utils as du
import estimator_pipeline as ep
import argparse

def evaluate_model(model_name, run_id):
    pat_idx, X, X_mask, y, y_mask, othr_data = ep.load_ichi_data(data_file=ep.DATA_file, is_padded=False)
    print("data loaded")
    yhat = du.load_base_model(ep.bam_loc, X, 5, [model_name,], [run_id,])
    yhat = yhat['model_' + model_name]
    print("model pred loaded")
    nRMSE = du.calculate_nRMSE(y, yhat, y_mask, X_mask)
    avg = np.nanmean(nRMSE)
    return nRMSE, avg

def main():
    arg = du.parse_args()
    model = arg.model
    run_id = arg.run
    nRMSE, avg = evaluate_model(model, run_id)
    print("Evaluation> {}/{} =>\nAverage:{}\nChannel:{}"
          .format(model, run_id, avg, nRMSE))

if __name__ == "__main__":
    main()