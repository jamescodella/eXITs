import pandas as pd
import numpy as np
import os
import sys
import pickle
import argparse
import data_utils as du

def data_to_hd5(train, input_path, output_path, d_types): # load individual CSV files and save as h5 file
    # read data files and create dataframes 
    for d_loc in d_types:
        num_files = len(os.listdir(input_path + d_loc))
        sys.stdout.write('Starting data load for: ' + d_loc + '\n')
        df = pd.DataFrame([])
        i = 1
        for files in os.listdir(input_path + d_loc):
            file = input_path + d_loc + '/' + files
            if file[-3:] == 'csv':
                temp = pd.read_csv(input_path + d_loc + '/' + files, header=0)
                temp['ID'] = files[:-4]
                df = df.append(temp)
                sys.stdout.write('\rLoading data... ' + str(i/num_files))
                sys.stdout.flush()
            i+=1
        sys.stdout.write('\nDone loading data at: ' + d_loc + '\n')
        df['ID'] = df['ID'].astype('int64')    
        df.to_hdf(output_path + d_loc + '.h5' ,key = d_loc)
    sys.stdout.write('Finished.')

def create_m(df, drop_columns = []): # create indicator matrix.  0- value is missing, 1- value is not missing
    drop_columns += ([c for c in df.columns if (('_m' in c) or ('_d' in c))])
    dft = 1-df.drop(drop_columns, axis=1).isna() 
    dft.columns = dft.columns+'_m'
    for i in drop_columns:
        dft[i] = df[i].copy()
    return dft

def calculate_diff(df): # Calculate time since last data point for each feature and each patient.
    drop_columns = ['ID','CHARTTIME']
    drop_columns += ([c for c in df.columns if (('_m' in c) or ('_d' in c))])
    df_diff = pd.DataFrame([])
    df_diff['ID'] = df['ID'].copy()
    df_diff['CHARTTIME'] = df['CHARTTIME'].copy()
    for target in df.drop(drop_columns,axis=1).columns: #include 'T' if calculate time_stamps
        t = df[['ID', 'CHARTTIME', target]]
        t = t.reset_index(drop=True)
        diff_name = target + "_d"
        t['diff'] = t.groupby('ID')['CHARTTIME'].diff()
        t['diff'] = t['diff'].fillna(0)
        t['c_diff']=t.groupby('ID')['diff'].cumsum()
        t['isnan'] = t[target].isnull()
        c = t['c_diff']
        c = c.fillna(0)
        t['time_diff'] = c.sub(c.mask(t['isnan'] != 0).ffill(), fill_value=0).astype(int)
        t[target+'_d'] = 0
        t.loc[t['time_diff'] == 0, diff_name] = t['time_diff'].shift(1) + t['diff']
        t.loc[t['time_diff'] != 0, diff_name] = t['time_diff']
        t.loc[t[diff_name] < 0, diff_name] = 0
        t[diff_name] = t[diff_name].fillna(0)
        t = t.drop(['diff','c_diff','isnan','time_diff'],axis=1)
        df_diff[diff_name] = t[diff_name]
    return df_diff 

def impute(df, strategy, w=[0.33, 0.33, 0.34]): # Impute missing values so sequences can be fed into an RNN. Can be forward fill, backfill, mean fill, or weighted average of all 3. 

    m_df = df.copy()
    r_df = df.copy()
    methods = ['ffill', 'bfill']
    if strategy in methods:
        r_df = df.fillna(method=strategy)
        methods.remove(strategy)
        r_df = r_df.fillna(method=methods[0]) 
    elif strategy in ['mean']:
        for col in df.columns:
            r_df[col] = df[col].fillna(df.groupby(['ID'])[col].transform('mean'))
    elif strategy in ['all']:
        for col in df.columns:
            m_df[col] = df[col].fillna(df.groupby(['ID'])[col].transform('mean'))
        df_f = df.fillna(method='ffill')         # Fill forward, then back
        df_f = df_f.fillna(method='bfill')
        df_b = df.fillna(method='bfill')         # Fill back, then forward
        df_b = df_b.fillna(method='ffill')
        r_df = w[0]*df_f + w[1]*df_b + w[2]*m_df
        r_df['ID'] = r_df['ID'].astype('int64')
        r_df['CHARTTIME'] = r_df['CHARTTIME'].astype('int64')
    return r_df

def create_sequences(df, drop_columns = ['ID','CHARTTIME']): # create sequences. dim (patients) (feature) (timestep)
    res = []
    dft = df.reset_index(drop=True)
    for i in dft['ID'].unique():
        temp_df = dft[dft['ID']==i]
        temp_df = temp_df.drop(drop_columns,axis=1)
        res.append(temp_df.values.T)
    return (res)

def pad_sequences(sequences, pad_value = -1): # Pads sequences
    maxlen = max([i.shape[1] for i in sequences])
    padded_seq=[]
    for i in range(len(sequences)):
        padded_seq.append(np.pad(sequences[i], pad_width=((0,0),(0,maxlen-sequences[i].shape[1]+1)),mode='constant', constant_values=pad_value))
    return padded_seq

def main():
    arg = du.parse_args()
    train = arg.train
    input_path = arg.inp
    output_path = arg.out
    if input_path == '': # specify default user paths here
        if train: 
            input_path = 'bam/challenge/data/'
        else:
            input_path = 'bam/challenge/'   
    if train:
        d_types = ['train_with_missing', 'train_groundtruth']
        dict_name = 'train_with_missing'
    else:
        d_types = ['test_with_missing']
        dict_name = 'test_with_missing'

    labTestNames = ['PCL', 'PK', 'PLCO2', 'PNA', 'HCT', 'HGB', 'MCV', 'PLT', 'WBC', 'RDW', 'PBUN', 'PCRE', 'PGLU']
    data_to_hd5(train, input_path = input_path, output_path = output_path, d_types=d_types)     # scan over all data files and save into hd5s
    df = {}
    for i in d_types:
        df[i] = {'data': pd.read_hdf(output_path + d_types[d_types.index(i)] + '.h5')}
        df[i]['data']['ID'] = df[i]['data']['ID'].astype('int64')
        df[i]['mask'] = create_m(df[i]['data'], drop_columns=['ID','CHARTTIME'])
        df[i]['diff'] = calculate_diff(df[i]['data'])
        df[i]['imputed_all'] = impute(df[i]['data'], 'all')
        print("Finished creating features for {} data...')".format(i))
        df[i]['sequences_data'] = sequences_df_m = create_sequences(df[i]['data']) # Training data with extra NaN values, scaled
        df[i]['sequences_mask'] = create_sequences(df[i]['mask']) # Mask of training data with extra NaN values,
        df[i]['sequences_diff'] = create_sequences(df[i]['diff']) # Time differences of training data with extra NaN values,
        df[i]['sequences_imputed_all'] = create_sequences(df[i]['imputed_all']) # Training data with extra NaN values, scaled
        keep_t = list(df[i]['data'].columns)
        keep_t.remove('CHARTTIME')
        df[i]['sequences_time'] = create_sequences(df[i]['data'], drop_columns=keep_t)
        df[i]['sequences_data_p'] = pad_sequences(df[i]['sequences_data'], pad_value=-1)
        df[i]['sequences_mask_p'] = pad_sequences(df[i]['sequences_mask'], pad_value=-1)
        df[i]['sequences_diff_p'] = pad_sequences(df[i]['sequences_diff'], pad_value=-1)
        df[i]['sequences_time_p'] = p = pad_sequences(df[i]['sequences_time'], pad_value=-1)
        df[i]['sequences_imputed_all_p'] = pad_sequences(df[i]['sequences_imputed_all'], pad_value=-1)
        df[i]['ID'] = df[i]['data']['ID'].drop_duplicates().values
        print('Squences created for {}.'.format(i))
        data = {'X': df[dict_name]['sequences_data'], 'X_filled' : df[dict_name]['sequences_imputed_all'], 'X_mask': df[dict_name]['sequences_mask'], 'X_diff': df[dict_name]['sequences_diff'], 'X_time': df[dict_name]['sequences_time'], 'ID':df[i]['ID']}
        data_p = {'X': df[dict_name]['sequences_data_p'], 'X_filled' : df[dict_name]['sequences_imputed_all_p'], 'X_mask': df[dict_name]['sequences_mask_p'], 'X_diff': df[dict_name]['sequences_diff_p'], 'X_time': df[dict_name]['sequences_time_p'],'ID':df[i]['ID']}
       
    if train:
        data.update({'y': df['train_groundtruth']['sequences_data'], 'y_mask': df['train_groundtruth']['sequences_mask'], 'y_filled': df['train_groundtruth']['sequences_imputed_all']})
        data_p.update({'y': df['train_groundtruth']['sequences_data_p'], 'y_mask': df['train_groundtruth']['sequences_mask_p'], 'y_filled': df['train_groundtruth']['sequences_imputed_all_p']})
        output_name = '_training_data.p'
    else:
        output_name = '_test_data.p'

    if train:
        labTestNames = ['PCL', 'PK', 'PLCO2', 'PNA', 'HCT', 'HGB', 'MCV', 'PLT', 'WBC', 'RDW', 'PBUN', 'PCRE', 'PGLU']
        samplesStatsMap = du.get_samples_statistics(data['X'], labTestNames)
        with open(output_path + 'stats.pkl', 'wb') as fp:
            pickle.dump(samplesStatsMap, fp)
            pickle.dump(labTestNames, fp)
    else:
        if not os.path.exists(output_path + 'stats.pkl'):
            print('Error: we need samples statistics')
        with open(output_path + 'stats.pkl', 'rb') as fp:
            samplesStatsMap = pickle.load(fp)
    data['X_interpolated_linear'] = du.interpolate_linear_x(data['X'], data['X_time'], samplesStatsMap)
    pickle.dump(data, open(output_path+"ICHI"+output_name, 'wb'))
    pickle.dump(data_p, open(output_path+"ICHI_PADDED"+output_name, 'wb'))
    print('Data for model inputs stored as .p files in {}'.format(output_path))
    return

if __name__ == "__main__":
    main()