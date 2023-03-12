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

model = load_model('model_roll_v4-169-fine-tune-20')
df_validation = pd.read_parquet('./v4/validation.parquet')
features = [c for c in df_validation if c.startswith("feature")]
X_val = df_validation[features].fillna(0.5).values
y_pred = model.predict(X_val, num_iteration=100)
valid_gen_and_save(y_pred, "roll_v4-169-fine-tune-20")
