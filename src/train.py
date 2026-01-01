
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import glob
import pickle
from pathlib import Path
import os

from sklearn.model_selection import train_test_split
import sklearn.metrics as confusion_matrix
from xgboost import XGBClassifier
from sklearn.metrics import classification_report,roc_auc_score,confusion_matrix,accuracy_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


directory = "model"
if not os.path.exists(directory):
    os.makedirs(directory)

#function to load and merge csv files for dataset
csv_filename = "combined_slice_dataset.csv"
def load_dataset(csv_filename):
    combined_csv_path = os.path.join(os.path.dirname(__file__), csv_filename)
            
    df = pd.read_csv(combined_csv_path)
    df = df.drop(df.filter(regex='^Unnamed').columns,axis=1)
    
    # Improved column cleaning logic
    df.columns = (df.columns
                  .str.replace('%', 'pct')
                  .str.replace(r'[\[\]\(\)]', '', regex=True)
                  .str.replace(' ', '_')
                  .str.lower())
    return df

df=load_dataset()
df=df.fillna(0)

#feature engineering 
#convert timestamp to network load
def convert_timestamp_to_networkLoad(timestamp):
    dt =pd.to_datetime(timestamp,unit='ms',utc=True).tz_convert("Europe/Rome")
    hour = dt.hour
    if hour >=17 and hour <=21:
        load = 'peak'
    elif hour >=0 and hour<7:
        load = 'night'
    else:
        load = 'off-peak'
    return hour,load

df[['hour','network_load']] = df['timestamp'].apply(convert_timestamp_to_networkLoad).apply(pd.Series)

column_names = [ 'timestamp','num_ues', 'imsi', 'rnti', 'slicing_enabled', 'slice_id',
       'slice_prb', 'power_multiplier', 'scheduling_policy', 'dl_mcs',
       'dl_n_samples', 'dl_buffer_bytes', 'tx_brate_downlink_mbps',
       'tx_pkts_downlink', 'tx_errors_downlink_pct', 'dl_cqi', 'ul_mcs',
       'ul_n_samples', 'ul_buffer_bytes', 'rx_brate_uplink_mbps',
       'rx_pkts_uplink', 'rx_errors_uplink_pct', 'ul_rssi', 'ul_sinr', 'phr',
       'sum_requested_prbs', 'sum_granted_prbs', 'dl_pmi', 'dl_ri', 'ul_n',
       'ul_turbo_iters','hour','network_load']


df_fulltrain , df_test= train_test_split(df,test_size=0.2,random_state=42)
df_train,df_val=train_test_split(df_fulltrain,test_size=0.25,random_state=42)
df_fulltrain = df_fulltrain.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)  
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


y_fulltrain = df_fulltrain['slice_id'].values
y_train = df_train['slice_id'].values
y_val = df_val['slice_id'].values
y_test = df_test['slice_id'].values

def print_classification_report(y_pred,y_pred_proba,actual_y):
    print(classification_report(actual_y, y_pred))
    print("*******************************************************")
    print(confusion_matrix(actual_y, y_pred))
    print("roc_auc score ",roc_auc_score(actual_y, y_pred_proba,multi_class="ovr"))

def remove_static_columns_from_dataset(column_names):
    for col in column_names:
        del df_train[col]
        del df_fulltrain[col]
        del df_val[col]
        del df_test[col]

#columns_to_keep = ['dl_mcs', 'ul_sinr', 'tx_brate_downlink_mbps', 'dl_buffer_bytes','ul_turbo_iters', 'dl_cqi',
     #  'ul_n_samples','network_load']

column_names = [ 'timestamp','num_ues', 'imsi', 'rnti', 'slicing_enabled', 'slice_id',
       'slice_prb', 'power_multiplier', 'scheduling_policy', 
         'dl_n_samples', 
        'tx_errors_downlink_pct', 'ul_mcs', 'ul_buffer_bytes', 'rx_brate_uplink_mbps',
       'rx_pkts_uplink', 'rx_errors_uplink_pct', 'ul_rssi',  'phr',
       'sum_requested_prbs', 'sum_granted_prbs', 'dl_pmi', 'dl_ri', 'ul_n','tx_pkts_downlink','hour'
       ]

remove_static_columns_from_dataset(column_names)

def train_model():

    pipeline =make_pipeline(DictVectorizer(sparse=False),XGBClassifier(
        n_estimators= 405,
        max_depth= 5,
        learning_rate= 0.179,
        subsample= 0.903,
        colsample_bytree= 0.716,
        min_child_weight= 4,
        gamma= 0.287,
        random_state=42
    ))

    pipeline.fit(df_fulltrain.to_dict(orient='records'),y_fulltrain)
    return pipeline

def save_model(pipeline,filename):
    with open(os.path.join(directory,filename),'wb') as f_out:
        pickle.dump(pipeline,f_out)
    print(f"Model is saved to {os.path.join(directory,filename)}")

pipeline = train_model()
save_model(pipeline,'xgboost.bin')

