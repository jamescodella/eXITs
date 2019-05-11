import os
import data_utils as du
import pickle
def main(arg):
    data_loc = arg.inp
    processed_data_loc = arg.out
    models = ['BiLSTM', 'BiGRU', 'HorizontalFcn', 'HorizontalKnn', 'HorizontalRandomForest', 'InterpolateLinear']
    folds = 5
    os.system("python process_data.py -t -i "+ data_loc + " -o " + processed_data_loc) # process training data
    os.system("python process_data.py -i "+ data_loc + " -o " + processed_data_loc) # process testing data
    for m in models: # Train and save each base model and their predictions
        print("training base model: ", m)
        os.system("python " + m + ".py -n 5 " )
        os.system("python " + m + ".py --final" )
    print("training eXITs...")
    os.system("python " + "ensemble_main.py -em XGB -n 5" )     # Train eXITs 
    os.system("python " + "ensemble_main.py -em XGB --final" )     # Generate eXITs prediction
    preds = pickle.load(open(processed_data_loc + 'XGB_0_Final', 'rb'))
    testdata = pickle.load(open(processed_data_loc + 'ICHI_test_data.p', 'rb'))
    os.makedirs(os.path.dirname(processed_data_loc+'output_results/'),exist_ok=True)   
    print("outputing predictions to CSV files in: " + processed_data_loc+'output/') 
    du.output_to_csv(preds['pred_y_test'], testdata['X'], testdata['X_mask'], testdata['X_time'], preds['othr_data']['ID'],processed_data_loc+'output_results/')

if __name__ == "__main__":
    arg = du.parse_args()
    main(arg)
