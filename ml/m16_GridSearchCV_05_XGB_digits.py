from sklearn.datasets import load_digits
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold
from sklearn.metrics import accuracy_score
import time
from xgboost import XGBRegressor, XGBRFClassifier

x, y = load_digits(return_X_y=True) # 사이킷런에서 사용 가능한 방식이다.

# print(x)
# print(y)
# print(x.shape, y.shape) #(1797, 64) (1797,)

print(pd.value_counts(y, sort=False))
# 0    178
# 1    182
# 2    177
# 3    183
# 4    181
# 5    182
# 6    181
# 7    179
# 8    174
# 9    180
# dtype: int64

y = pd.get_dummies(y)
print(y)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1346, train_size=0.75, shuffle=True, stratify=y)

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
    {'n_jobs' : [-1,], 'n_estimators' : [100, 500], 'max_depth' : [6, 10, 12],
     'min_samples_leaf' : [3, 10]},
    {'n_jobs' : [-1,], 'max_depth' : [6, 7, 10, 12],
     'min_samples_leaf' : [3, 5, 7, 10]},
    {'n_jobs' : [-1,], 'min_samples_leaf' : [3, 5, 7, 10],
     'min_samples_split' : [2, 3, 5, 10]},   
    {'n_jobs' : [-1,], 'min_samples_split' : [2, 3, 5, 10]},   
]


#2. 모델
model = GridSearchCV(XGBRFClassifier(), parameters, cv=kfold,
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
print('accuracy_score :', accuracy_score(y_test,y_pred))

y_pred_best = model.best_estimator_.predict(x_test)
print('accuracy_score :', accuracy_score(y_test, y_pred_best))

print('걸린 시간 :', round(end_time - start_time, 2), '초')

# 최적의 매개변수 : XGBRFClassifier(base_score=None, booster=None, callbacks=None,
#                 colsample_bylevel=None, colsample_bytree=None, device=None,
#                 early_stopping_rounds=None, enable_categorical=False,
#                 eval_metric=None, feature_types=None, gamma=None,
#                 grow_policy=None, importance_type=None,
#                 interaction_constraints=None, max_bin=None,
#                 max_cat_threshold=None, max_cat_to_onehot=None,
#                 max_delta_step=None, max_depth=7, max_leaves=None,
#                 min_child_weight=None, min_samples_leaf=3, missing=nan,
                # monotone_constraints=None, multi_strategy=None,
                # n_estimators=None, n_jobs=-1, num_parallel_tree=None,
                # objective='binary:logistic', random_state=None, ...)
# 최적의 파라미터 {'max_depth': 7, 'min_samples_leaf': 3, 'n_jobs': -1}
# best_score :  0.8091973013906101
# model.score : 0.8355555555555556
# accuracy_score : 0.8355555555555556
# accuracy_score : 0.8355555555555556
# 걸린 시간 : 11.96 초