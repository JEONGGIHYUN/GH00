from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import time
from xgboost import XGBRegressor, XGBRFClassifier

#1. 데이터
datasets = load_diabetes()

x = datasets.data
y = datasets.target

# print(x.shape, y.shape) (442, 10) (442, )

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=7251)
################################################
from sklearn.preprocessing import MinMaxScaler , StandardScaler
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
# 걸린 시간 : 4.42 초


#   warnings.warn(smsg, UserWarning)
# 최적의 매개변수 : XGBRegressor(base_score=None, booster=None, callbacks=None,
#              colsample_bylevel=None, colsample_bynode=None,
#              colsample_bytree=None, device=None, early_stopping_rounds=None,
#              enable_categorical=False, eval_metric=None, feature_types=None,
#              gamma=None, grow_policy=None, importance_type=None,
#              interaction_constraints=None, learning_rate=None, max_bin=None,
#              max_cat_threshold=None, max_cat_to_onehot=None,
#              max_delta_step=None, max_depth=None, max_leaves=None,
#              min_child_weight=None, min_samples_leaf=5, min_samples_split=10,
#              missing=nan, monotone_constraints=None, multi_strategy=None,
#              n_estimators=None, n_jobs=-1, ...)
# 최적의 파라미터 {'running_rate': 0.1, 'n_jobs': -1, 'min_samples_split': 10, 'min_samples_leaf': 5}
# best_score :  0.2562370539733756
# model.score : 0.428190642533554
# accuracy_score : 0.428190642533554
# accuracy_score : 0.428190642533554
# 걸린 시간 : 3.08 초