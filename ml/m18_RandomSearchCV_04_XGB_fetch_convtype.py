import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import time
from xgboost import XGBRegressor, XGBRFClassifier


#1. 데이터
datasets = fetch_covtype()
# print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data # (581012, 54)
y = datasets.target # (581012,)
# print(x.shape, y.shape)

# print(np.unique(y, return_counts=True))
#  (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],
    #   dtype=int64))

# print(pd.value_counts(y))
# 2    283301
# 1    211840
# 3     35754
# 7     20510
# 6     17367
# 5      9493
# 4      2747
# dtype: int64

# y = pd.get_dummies(y)
# print(y)
# print(y.shape)

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape)

# from sklearn.preprocessing import OneHotEncoder
# y_ohe3 = y.reshape(-1, 1)
# y_ohe = OneHotEncoder(sparse=False) #True가 기본값
# y_ohe3 = y_ohe.fit_transform(y_ohe3)
# print(y_ohe3)
# print(y_ohe3.shape)


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1356, train_size=0.75, shuffle=True, stratify=y)

###############################################
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
scaler = RobustScaler()

scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)
################################################

n_splits = 5 
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)

parameters =[
    {'n_jobs' : [-1,], 'n_estimators' : [100, 500], 'max_depth' : [6, 10, 12], 'running_rate' : [0.1, 0.01, 0.001, 0.0001],
     'min_samples_leaf' : [3, 10]},
    {'n_jobs' : [-1,], 'max_depth' : [6, 7, 10, 12],'running_rate' : [0.1, 0.01, 0.001, 0.0001],
     'min_samples_leaf' : [3, 5, 7, 10]},
    {'n_jobs' : [-1,], 'min_samples_leaf' : [3, 5, 7, 10],'running_rate' : [0.1, 0.01, 0.001, 0.0001],
     'min_samples_split' : [2, 3, 5, 10]},   
    {'n_jobs' : [-1,], 'min_samples_split' : [2, 3, 5, 10]},   
]



#2. 모델
model = RandomizedSearchCV(XGBRegressor(), parameters, cv=kfold,
                     verbose=1,
                     refit=True,
                     n_jobs=-1,
                     n_iter=10
                     )



#3. 훈련
start_time = time.time()
model.fit(x_train, y_train,
          eval_set = [(x_train, y_train), (x_test, y_test)],
          verbose = True)
end_time = time.time()

#4. 예측
print('최적의 매개변수 :', model.best_estimator_)

print('최적의 파라미터',model.best_params_)

print('best_score : ', model.best_score_)

print('model.score :', model.score(x_test,y_test))

y_pred = model.predict(x_test)
print('accuracy_score :', accuracy_score(y_test,y_pred))

y_pred_best = model.best_estimator_.predict(x_test)
print('accuracy_score :', accuracy_score(y_test, y_pred_best))

print('걸린 시간 :', round(end_time - start_time, 2), '초')

# 최적의 매개변수 : XGBRegressor(base_score=None, booster=None, callbacks=None,
#              colsample_bylevel=None, colsample_bynode=None,
#              colsample_bytree=None, device=None, early_stopping_rounds=None,
#              enable_categorical=False, eval_metric=None, feature_types=None,
#              gamma=None, grow_policy=None, importance_type=None,
#              interaction_constraints=None, learning_rate=None, max_bin=None,
#              max_cat_threshold=None, max_cat_to_onehot=None,
#              max_delta_step=None, max_depth=12, max_leaves=None,
#              min_child_weight=None, min_samples_leaf=3, missing=nan,
#              monotone_constraints=None, multi_strategy=None, n_estimators=500,
#              n_jobs=-1, num_parallel_tree=None, ...)
# 최적의 파라미터 {'running_rate': 0.1, 'n_jobs': -1, 'n_estimators': 500, 'min_samples_leaf': 3, 'max_depth': 12}
# best_score :  0.8822998881340027
# model.score : 0.8915883302688599
# Traceback (most recent call last):
#   File "c:\ai5\study\ml\m17_RandomSearchCV_04_XGB_fetch_convtype.py", line 111, in <module>
#     print('accuracy_score :', accuracy_score(y_test,y_pred))
#   File "c:\Users\tds\AppData\Local\anaconda3\envs\tf274gpu\lib\site-packages\sklearn\utils\_param_validation.py", line 213, in wrapper
#     return func(*args, **kwargs)
#   File "c:\Users\tds\AppData\Local\anaconda3\envs\tf274gpu\lib\site-packages\sklearn\metrics\_classification.py", line 231, in accuracy_score
#     y_type, y_true, y_pred = _check_targets(y_true, y_pred)
#   File "c:\Users\tds\AppData\Local\anaconda3\envs\tf274gpu\lib\site-packages\sklearn\metrics\_classification.py", line 112, in _check_targets
#     raise ValueError(
# ValueError: Classification metrics can't handle a mix of multilabel-indicator and continuous-multioutput targets

