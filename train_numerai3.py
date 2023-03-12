import matplotlib
import gc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import random
import sklearn
import lightgbm
import matplotlib.pyplot as plt
import seaborn as sns
import re
import joblib
import lightgbm as lgb
import math
from collections import Counter
from numerapi import NumerAPI
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ARDRegression
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from utils import save_model, load_model, neutralize, get_biggest_change_features, validation_metrics
from sklearn import (
    feature_extraction, feature_selection, decomposition, linear_model,
    model_selection, metrics, svm
)
from utils import (
    save_model, load_model, neutralize,
    get_biggest_change_features,
    get_time_series_cross_val_splits,
    validation_metrics,
    ERA_COL, DATA_TYPE_COL, TARGET_COL, EXAMPLE_PREDS_COL
)

def prediction_clip(read_file, save_name):
    read = pd.read_csv(read_file)
    read.loc[read['prediction'] < 0, 'prediction'] = 0
    read.loc[read['prediction'] > 1, 'prediction'] = 1
    read.to_csv(read_file, index=False)

#input np.array of the prediction from the model
def valid_gen_and_save(prediction,version_name):
    id = df_validation.index
    data = {'id': id, 'prediction': np.round(prediction, 8)}
    pred_file = pd.DataFrame(data)
    current_round = 433
    file_name = f"./prediction/val_{current_round}_{version_name}.csv"
    pred_file.to_csv(file_name, index=False)
    prediction_clip(file_name, file_name)

df_train = pd.read_parquet('./v4/train.parquet')
df_train.info()

features = [c for c in df_train if c.startswith("feature")]
targets = [c for c in df_train if c.startswith("target")]
targets_60 = [c for c in df_train if (c.startswith("target") and c.endswith("60"))]
eras_features_and_targets = [c for c in df_train if (c.startswith("target") or c.startswith("feature") or c.startswith("era"))]
feature_size = len(features)
target_size = len(targets)
eras_features_and_targets_size = len(eras_features_and_targets)

print("The size of the feature is:", feature_size)
print("The size of the targets is:", target_size)
print("The size of eras + features + targets =", eras_features_and_targets_size)

X_train = df_train
X_train['era'] = X_train['era'].astype(int)
eras = df_train['era'].unique()
eras_len = len(eras)


######################## by 1 era ######################
timeSeries = np.array([i for i in range(1,eras_len+1)])
tscv = TimeSeriesSplit(gap=1, max_train_size=1, n_splits=572, test_size=None)

for i, (train_index, test_index) in enumerate(tscv.split(timeSeries)):

    print(f"Fold {i+1}:")
    train_index += 1
    test_index += 1
    train_index = train_index.tolist()
    test_index = test_index.tolist()

    print(f"  Train: index={set(train_index)}")
    print(f"  Test:  index={set(test_index)}")
        
    mask_train = X_train['era'].isin(set(train_index))
    mask_val = X_train['era'].isin(set(test_index))

    X_era = X_train[mask_train][features].fillna(0.5).values
    y_era = X_train[mask_train]['target'].fillna(0.5).values
    X_test = X_train[mask_val][features].fillna(0.5).values
    y_test = X_train[mask_val]['target'].fillna(0.5).values


    num_rounds = 50000
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'learning_rate': 0.01,
        'early_stopping': 50,
        'verbose': -1,
    }
    train_data = lgb.Dataset(X_era, label=y_era)
    valid_data = lgb.Dataset(X_test, label=y_test)
    model = lgb.train(params, train_data, num_rounds, valid_sets=[train_data, valid_data])
    joblib.dump(model, './models/v4_kFold_era1_r4.pkl')


df_validation = pd.read_parquet('./v4/validation.parquet')
X_val = df_validation[features].fillna(0.5).values
y_pred = model.predict(X_val, num_iteration=100)
valid_gen_and_save(y_pred,"v4_kFold_era1_r4")




######################## by 10 groups ######################

kGroup = [10,20,50]


for k in kGroup:

    timeSeries = np.array([i for i in range(1,eras_len+1)])
    tscv = TimeSeriesSplit(gap=1, max_train_size=none,  n_splits=k, test_size=None)

    for i, (train_index, test_index) in enumerate(tscv.split(timeSeries)):

        print(f"Fold {i+1}:")
        train_index += 1
        test_index += 1
        train_index = train_index.tolist()
        test_index = test_index.tolist()

        print(f"  Train: index={set(train_index)}")
        print(f"  Test:  index={set(test_index)}")
            
        mask_train = X_train['era'].isin(set(train_index))
        mask_val = X_train['era'].isin(set(test_index))

        X_era = X_train[mask_train][features].fillna(0.5).values
        y_era = X_train[mask_train]['target'].fillna(0.5).values
        X_test = X_train[mask_val][features].fillna(0.5).values
        y_test = X_train[mask_val]['target'].fillna(0.5).values


        num_rounds = 50000
        params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'learning_rate': 0.01,
            'early_stopping': 50,
            'verbose': -1,
        }
        train_data = lgb.Dataset(X_era, label=y_era)
        valid_data = lgb.Dataset(X_test, label=y_test)
        model = lgb.train(params, train_data, num_rounds, valid_sets=[train_data, valid_data])
        joblib.dump(model, './models/v4_kFold_'+k+'_r4.pkl')

    df_validation = pd.read_parquet('./v4/validation.parquet')
    X_val = df_validation[features].fillna(0.5).values
    y_pred = model.predict(X_val, num_iteration=100)
    valid_gen_and_save(y_pred,"v4_kFold_"+k+"_r4")

