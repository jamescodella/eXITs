# eXtreme Imputation: An ensemble approach for imputing missing data

This repo contains the code for our IBM Research team's entry in the IEEE International Conference on Health Informatics Data Analytics Challenge on Missing data Imputation (DACMI). The code is copyright 2019 IBM, Corp and open sourced under the Apache License Version 2.0 (APLv2). [See license.txt]

The authors of this code (in alphabetical order) are: Prithwish Chakraborty, James Codella, Mohamed Ghalwash, Hillol Sarker, Daby Sow, and Zijun Yao

eXITs requires Python 3.6

# Instructions

Please follow the steps listed below to process the ICHI DACMI data, train and evaluate models, and generate predictions.

## Automated training and prediction generation

The `run.py` script will process the data files, train the base models and eXITs, and generate predictions. Processed data is saved to `bam/data`, models are saved to `bam/models/`.  Imputations will be stored in `bam/data/output_results/`
Run `python run.py -i <input_path> ` where
```
- input_path: path must contain "train_with_missing", "train_groundtruth", and "test_missing" folders each containing the corresponding data CSVs.
```
E.g. `python run.py -i bam/challenge/`

## Manual/Custom Training, Evaluation

**Please note:**
1. Before running any models, you need to process the dataset and specify the dataset location
2. Before evaluating **base models**, you need to train on all five of the cross-validation folds (configurations 0-4) and on the entire set.
3. Before evaluating  **ensembles**, you need to train on all five of the cross-validation folds (configurations 0-4) and on the entire set.


## 0. Process data
Run `python process_data.py -i </path_to_data/> -o </path_to_processed_data/>` and use the flag  `-t` to process the training set, or omit `-t` to process the test set.
E.g. `python process_data.py -i models/bam/data/ -o models/bam/data - t` processes the training data

## 1. Base models
You need to train the base models on each cross-validation fold. There are five folds in total (0-4). 
Before moving on to train the ensemble model, you need to train each base model in the ensemble on each fold, then on once on data from all the folds.

### 1.1 Training
To train a model, run `MODEL_NAME.py -n <narg>`
where narg =  0-4 runs Cross Validation on different fold configuration, and narg = 5 trains on the entire data set minus the dev set
Example: `../models/python BiLSTM.py -n 0`

To train the model on the all the folds (i.e. the entire train set minus dev set), use the arg `-n 5`
Example: `../models/python BiLSTM.py -n 5`

### 1.2 Evaluation
To evaluate a base model and compute the average and feature-wise nRMSE, run eval.py on the base model name. 
Example: `../models/python eval.py -m BiLSTM`

### 1.3 Prediction
To make predictions on the test data, run the model code with the `--final` argument
Example: `../models/python BiLSTM.py --final`

## 2 Ensemble 
After training the base models on folds 0 through 4, and on the entire training set (i.e. `-n 5`). Then you can start training the ensemble model.

### 2.1 Configuration
The configuration file for the ensembler is in `../models/ensemble/config.yml` and allows the user to configure:
```
- standard_ensembles: the list of ensemble files to choose as an argument
- gate_ensemble: the structure and parameters of the gating ensembler 
- feedforward_ensemble: the structure and parameters of the feed-forward network ensembler 
- base_models: A list of base models to be used by an ensemble.
```

### 2.2 Training
Training the ensemble can be done by running `ensemble_main.py -em <enemble_name>, -n <narg> -t` where:
```
  - ensemble_name = name of ensemble model as listed in `standard_ensembles` in config.yml.
  - narg =  0-4 runs Cross Validation on different fold configuration, and narg = 5 trains on the entire data set minus the dev set
  - t = flag to run ensemble training. If not specified, ensemble will predict on the data.
``` 
Example: `../models/python ensemble_main.py -em XGB -n 0 -t `

### 2.3 Evaluation
To evaluate the ensemble and compute the average and per-feature nRMSE, run eval.py on the trained ensemble model name. 
Example: `../models/python eval.py -em XGB`

### 2.4 Generate predictions
To generate a predictions using the ensemble, run ensemble_main.py on the model and give the `-f` arg
Example: `../models/python ensemble_main.py -em XGB -f`
