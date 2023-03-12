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
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
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
# %matplotlib inline

print("acc")
napi = NumerAPI()
print("Downloading dataset files...")
dataset_name = "v4"
# feature_set_name = "medium"
napi = NumerAPI()
current_round = napi.get_current_round()
# napi.download_dataset("{dataset_name}/train.parquet", "train.parquet")
# napi.download_dataset("{dataset_name}/validation.parquet", "validation.parquet")
# napi.download_dataset("{dataset_name}/live.parquet", "live.parquet")
# napi.download_dataset("{dataset_name}/live_example_preds.parquet", "live_example_preds.parquet")
# napi.download_dataset("{dataset_name}/validation_example_preds.parquet", "validation_example_preds.parquet")
# napi.download_dataset("{dataset_name}/features.json", "features.json")
# napi.download_dataset("{dataset_name}/meta_model.parquet", "meta_model.parquet")

# napi.download_dataset()
# napi.download_dataset(
#     f"{dataset_name}/live_int8.parquet",
#     f"{dataset_name}/live_int8_{current_round}.parquet",
#     f"{dataset_name}/train_int8.parquet"
# )

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

print("readiong datasets...")
df_train = pd.read_parquet('./v4/train.parquet')
df_validation = pd.read_parquet('./v4/validation.parquet')
df_live = pd.read_parquet('./v4/live.parquet')
df_train.info()
df_validation.info()
df_live.info()

features = [c for c in df_train if c.startswith("feature")]
targets = [c for c in df_train if c.startswith("target")]
eras_features_and_targets = [c for c in df_train if (c.startswith("target") or c.startswith("feature") or c.startswith("era"))]
feature_size = len(features)
target_size = len(targets)
eras_features_and_targets_size = len(eras_features_and_targets)

print("The size of the feature is:", feature_size)
print("The size of the targets is:", target_size)
print("The size of eras + features + targets =", eras_features_and_targets_size)


eras = df_train['era'].unique()
X_val = df_validation[features].fillna(0.5).values
X = df_train
X['era'] = X['era'].astype(int)
from sklearn.model_selection import TimeSeriesSplit
eras = df_train['era'].unique()
X_val = df_validation[features].fillna(0.5).values
X = df_train
X['era'] = X['era'].astype(int)
# int_model = './models/model_roll_v2.txt'
start_era = 331

for era in list(range(start_era,len(eras)+1),12):
    
    window = list(range(1,era+1))
    num_rounds = int(10 + math.sqrt(era)*20)             #iteration
    lr = 1/math.sqrt(era)
    mask = X['era'].isin(window)
    mask_stratify = (X[mask]['era'] == era)
    
    X_era = X[mask][features].fillna(0.5).values
    y_era = X[mask].target.fillna(0.5).values
    X_train, X_test, y_train, y_test = train_test_split(X_era, y_era, test_size=0.3, stratify=mask_stratify, random_state=42)
    
    print(f'round era = {era}; iteration: {num_rounds}; learning rate: {lr}')   
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test)
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mse',
        'learning_rate': lr,
        'max_depth': 5,
        # 'num_leaves': 2^5,
        # 'num_threads': 24,
        'verbose': 0,
        'lambda_l2': 0.3,
        # 'device': 'gpu'
    }

    if era == 1:
        model = lgb.train(params, train_data, num_rounds, valid_sets=[valid_data])
    else:
        model = lgb.train(params, train_data, num_rounds, valid_sets=[valid_data], init_model = './models/model_roll_v2.txt')
    
    print("saving model...")
    y_pred = model.predict(X_val, num_iteration=100)
    model.save_model('./models/model_roll_v2.txt')

    #save prediction every 10 iterations
    if era%10 ==0:
        valid_gen_and_save(y_pred,f"time_v2_era{era}")

y_pred = model.predict(X_val, num_iteration=100)
   