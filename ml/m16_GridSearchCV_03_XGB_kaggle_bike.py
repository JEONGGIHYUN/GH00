# https://www.kaggle.com/competitions/bike-sharing-demand/data

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold
from sklearn.metrics import accuracy_score
import time
from xgboost import XGBRegressor, XGBRFClassifier


#1. 데이터
# path = './_data/bike_sharing_demand/' # 상대 경로 방식

# * 3개 다 같은 방식이다 *
# path = 'C:\\TDS\\ai5\\_data\\bike-sharing-demand' # 절대 경로 방식
# path = 'C://TDS//ai5//_data//bike-sharing-demand//' 
path = 'C:/ai5/_data/bike-sharing-demand/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
sampleSubmission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)

print(train_csv.shape) # (10886, 11)
print(test_csv.shape) # (6493, 8)
print(sampleSubmission.shape) # (6493, 1)

print(train_csv.columns)

print(train_csv.info())
print(test_csv.info())

print(train_csv.describe()) 

######### 결측치 확인 #########
print(train_csv.isnull().sum())
print(train_csv.isna().sum())

print(test_csv.isnull().sum())
print(test_csv.isna().sum())

###### x와 y를 분리 ########
x = train_csv.drop(['casual','registered','count'], axis=1)
print(x.shape) # (10886, 8)
# print(x)

y = train_csv['count']
print(y.shape)

# print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.87, random_state=57543)

################################################
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
    {'n_jobs' : [-1,], 'n_estimators' : [100, 500], 'max_depth' : [6, 10, 12],
     'min_samples_leaf' : [3, 10]},
    {'n_jobs' : [-1,], 'max_depth' : [6, 7, 10, 12],
     'min_samples_leaf' : [3, 5, 7, 10]},
    {'n_jobs' : [-1,], 'min_samples_leaf' : [3, 5, 7, 10],
     'min_samples_split' : [2, 3, 5, 10]},   
    {'n_jobs' : [-1,], 'min_samples_split' : [2, 3, 5, 10]},   
]


#2. 모델
model = GridSearchCV(XGBRegressor(), parameters, cv=kfold,
                     verbose=1,
                     refit=True,
                     n_jobs=-1,
                     )

#3. 훈련
start_time = time.time()
model.fit(x_train,y_train)
end_time = time.time()

#4. 예측
print('최적의 매개변수 :', model.best_estimator_)

print('최적의 파라미터',model.best_params_)

print('best_score : ', model.best_score_)

print('model.score :', model.score(x_test,y_test))

y_pred = model.predict(x_test)
print('accuracy_score :', r2_score(y_test,y_pred))

y_pred_best = model.best_estimator_.predict(x_test)
print('accuracy_score :', r2_score(y_test, y_pred_best))

print('걸린 시간 :', round(end_time - start_time, 2), '초')

# Parameters: { "min_samples_leaf" } are not used.

#   warnings.warn(smsg, UserWarning)
# 최적의 매개변수 : XGBRegressor(base_score=None, booster=None, callbacks=None,
#              colsample_bylevel=None, colsample_bynode=None,
#              colsample_bytree=None, device=None, early_stopping_rounds=None,
#              enable_categorical=False, eval_metric=None, feature_types=None,
#              gamma=None, grow_policy=None, importance_type=None,
#              interaction_constraints=None, learning_rate=None, max_bin=None,
#              max_cat_threshold=None, max_cat_to_onehot=None,
#              max_delta_step=None, max_depth=6, max_leaves=None,
#              min_child_weight=None, min_samples_leaf=3, missing=nan,
#              monotone_constraints=None, multi_strategy=None, n_estimators=100,
#              n_jobs=-1, num_parallel_tree=None, ...)
# 최적의 파라미터 {'max_depth': 6, 'min_samples_leaf': 3, 'n_estimators': 100, 'n_jobs': -1}
# best_score :  0.2562370539733756
# model.score : 0.428190642533554
# accuracy_score : 0.428190642533554
# accuracy_score : 0.428190642533554
# 걸린 시간 : 4.54 초