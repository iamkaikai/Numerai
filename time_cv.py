# import tsxv
# from tsxv.splitTrain import split_train
import sklearn
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

timeSeries = np.array([i for i in range(1,574+1)])
groups = 2
# groups = 10
trainSize = None
testSize = None
tscv = TimeSeriesSplit(gap=0, max_train_size=None, n_splits=groups, test_size=testSize)
# X, y = split_train(timeSeries, n_steps_input=4, n_steps_forecast=3, n_steps_jump=2)
# print(f"X = {X}")
# print(f"y = {y}")

for i, (train_index, test_index) in enumerate(tscv.split(timeSeries)):
    print(f"Fold {i+1}:")
    train_index += 1
    test_index += 1
    train_index = train_index.tolist()
    test_index = test_index.tolist()
    print(f"  Train: len={len(train_index)}")
    print(f"  Train: index={set(train_index)}")
    print(f"  Test:  index={set(test_index)}")