import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV


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

rf=RandomForestRegressor(n_estimators=80,  max_depth=8, min_samples_leaf=3, max_features='sqrt',random_state=110,n_jobs=-1)
param={"n_estimators":list(range(168,180)),"max_depth":[8,10]}
gc=GridSearchCV(rf,param,cv=2)
gc.fit(train_X, train_Y)

ptrain = gc.predict(train_X)
TRAIN = pd.concat([pd.DataFrame(train_Y), pd.DataFrame(ptrain)], axis=1)

ea_train, mse_train, mae_train, rmse_train, r2_train = cal_score(train_Y, ptrain)
print("建模精度：", ea_train, mse_train, mae_train, rmse_train, r2_train)


ptest = gc.predict(test_X)
TEST = pd.concat([pd.DataFrame(test_Y), pd.DataFrame(ptest)], axis=1)

ea_test, mse_test, mae_test, rmse_test, r2_test = cal_score(test_Y, ptest)
print("预测精度：", ea_test, mse_test, mae_test, rmse_test, r2_test)
print(pd.concat([TRAIN,TEST], axis=0))
print("测试集准确率：",gc.score(test_X,test_Y))
print("最好模型：",gc.best_estimator_)


Results = pd.concat([TRAIN,TEST], axis=0)

