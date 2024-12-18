# https://www.kaggle.com/c/otto-group-product-classification-challenge/leaderboard?tab=public
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder
import time
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import time
from xgboost import XGBRegressor, XGBRFClassifier
import xgboost as xgb

path = 'C:/ai5/_data/kaggle/otto-group-product-classification-challenge/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)

test_csv = pd.read_csv(path + 'test.csv', index_col=0)

submission_csv = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)

# train_csv.info()
# test_csv.info()
# print(train_csv['target'].value_counts())
# train_csv['target'] = train_csv['target'].replace({'Class_1' : 1, 'Class_1' : 1, 'Class_2' : 2, 'Class_3' : 3, 'Class_4' : 4, 'Class_5' : 5, 'Class_6' : 6, 'Class_7' : 7, 'Class_8' : 8, 'Class_9' : 9, })


############################################
x = train_csv.drop(['target'], axis=1)
# print(x.shape) # (61878, 93)

y = train_csv['target']
# print(y.shape) # (61878,)
###################################
y = pd.get_dummies(y)
# print(y)
# print(y.shape)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=3, shuffle=True, stratify=y)

###############################################
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
scaler = RobustScaler()

# scaler.fit(x_train)
# scaler.transform(x_train)
x_train = scaler.fit_transform(x_train)
scaler.transform(x_test)

###############################################

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
model = RandomizedSearchCV(XGBRFClassifier(), parameters, cv=kfold,
                     verbose=1,
                     refit=True,
                     n_jobs=-1,
                     n_iter=10
                     )



#3. 훈련
start_time = time.time()
model.fit(x_train, y_train,
          # eval_set = [(x_train, y_train), (x_test, y_test)],
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

#   warnings.warn(smsg, UserWarning)
# 최적의 매개변수 : XGBRFClassifier(base_score=None, booster=None, callbacks=None,
#                 colsample_bylevel=None, colsample_bytree=None, device=None,
#                 early_stopping_rounds=None, enable_categorical=False,
#                 eval_metric=None, feature_types=None, gamma=None,
#                 grow_policy=None, importance_type=None,
#                 interaction_constraints=None, max_bin=None,
#                 max_cat_threshold=None, max_cat_to_onehot=None,
#                 max_delta_step=None, max_depth=12, max_leaves=None,
#                 min_child_weight=None, min_samples_leaf=10, missing=nan,
#                 monotone_constraints=None, multi_strategy=None,
#                 n_estimators=500, n_jobs=-1, num_parallel_tree=None,
#                 objective='binary:logistic', random_state=None, ...)
# 최적의 파라미터 {'running_rate': 0.1, 'n_jobs': -1, 'n_estimators': 500, 'min_samples_leaf': 10, 'max_depth': 12}
# best_score :  0.6797803342175591
# model.score : 0.6629497953027365
# accuracy_score : 0.6629497953027365
# accuracy_score : 0.6629497953027365
# 걸린 시간 : 246.6 초

