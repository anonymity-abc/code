import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error, r2_score
from matplotlib import pyplot as plt
import scipy.io as scio
import warnings
from sklearn.ensemble import RandomForestRegressor
import csv
#import gdal
from sklearn import ensemble
from skimage import io
from osgeo import gdal, gdalconst
import os
from osgeo import gdal
import datetime
import shutil
import zipfile
from os.path import basename
import sys
from osgeo import gdal, gdal_array
warnings.filterwarnings("ignore")


def cal_score(y_true, y_pred):
    ev = explained_variance_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return ev, mse, mae, rmse, r2


data_train = pd.read_csv(r'F:\lai\1\jm.csv')
data_test = pd.read_csv(r'F:\lai\1\yz.csv')

train_X = data_train.iloc[:, 1:].values
train_Y = data_train.iloc[:, 0].values


test_X = data_test.iloc[:, 1:].values
test_Y = data_test.iloc[:, 0].values

reg_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=9,  min_samples_leaf=11, max_features= 'sqrt',random_state=100)
reg_model.fit(train_X, train_Y)


ptrain = reg_model.predict(train_X)
TRAIN = pd.concat([pd.DataFrame(train_Y), pd.DataFrame(ptrain) ], axis=1)

ea_train, mse_train, mae_train, rmse_train, r2_train = cal_score(train_Y, ptrain)
print("建模精度：", ea_train, mse_train, mae_train, rmse_train, r2_train)

ptest = reg_model.predict(test_X)
TEST = pd.concat([pd.DataFrame(test_Y), pd.DataFrame(ptest) ], axis=1)

ea_test, mse_test, mae_test, rmse_test, r2_test = cal_score(test_Y, ptest)
print("预测精度：", ea_test, mse_test, mae_test, rmse_test, r2_test)

Results = pd.concat([TRAIN,TEST], axis=0)

