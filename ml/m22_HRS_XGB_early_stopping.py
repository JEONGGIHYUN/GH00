# import numpy as np
# from sklearn.datasets import load_iris, load_diabetes
# from sklearn.model_selection import train_test_split, KFold
# from sklearn.svm import SVC
# from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
# from sklearn.experimental import enable_halving_search_cv
# from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
# from sklearn.metrics import accuracy_score
# from xgboost import XGBClassifier, XGBRegressor
# import time
# import warnings
# warnings.filterwarnings('ignore')


# #1. 데이터
# x, y = load_diabetes(return_X_y=True)

# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, shuffle=True, train_size=0.8)

# print(x_train.shape, y_train.shape) # (353, 10) (353,)
# print(x_test.shape, y_test.shape) # (89, 10) (89,)


# n_splits = 5 
# kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=333)

# parameters = [
#     {'learning_rate':[0.01, 0.05, 0.1, 0.2, 0.3], 
#      'gamma':[0, 0.1, 0.2, 0.5, 1.0], 
#      'max_depth' : [6, 7, 8, 9, 1], 
#     #  'subsample' : [0.6, 0.7, 0.8, 0.9, 1.0], 
#     #  'colsample_bytree' : [0.6, 0.7, 0.8, 0.9, 1.0], 
#      }

# ]

# import xgboost as xgb
# early_stop = xgb.callback.EarlyStopping(
#     rounds=50,
#     metric_name='mlogloss',
#     data_name='validation_0',
#     save_best=True
# )


# #2. 모델
# model = HalvingRandomSearchCV(XGBRegressor(
#                                           # tree_method='gpu_hist'
#                                           tree_method='hist',
#                                           device='cuda',
#                                           n_estimators=50,
#                                           eval_metric='mlogloss',
#                                           callbacks=[early_stop]
#                                          ),
#                      parameters,
#                      cv=kfold,
#                      verbose=1, # 1:이터레이터 내용만 2이상은 훈련내용도 같이 출력
#                      refit=True,
#                     #  n_jobs=-1,
#                     #  n_iter=10,
#                      min_resources=30,
#                      max_resources=353,
#                      random_state=4325,
#                      factor=3,
#                      aggressive_elimination=True,
#                      )

# #3. 훈련
# start_time = time.time()
# model.fit(x_train,y_train,
#           eval_set=[(x_test, y_test)],
#           verbose=True)
# end_time = time.time()

# #4. 예측
# print('최적의 매개변수 :', model.best_estimator_)

# print('최적의 파라미터',model.best_params_)

# print('best_score : ', model.best_score_)

# print('model.score :', model.score(x_test,y_test))

# y_pred = model.predict(x_test)
# print('accuracy_score :', accuracy_score(y_test,y_pred))

# y_pred_best = model.best_estimator_.predict(x_test)
# print('accuracy_score :', accuracy_score(y_test, y_pred_best))

# print('걸린 시간 :', round(end_time - start_time, 2), '초')

import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import time
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=15426, shuffle=True, train_size=0.8, stratify=y)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


n_splits = 5 
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=333)

parameters = [
    {'learning_rate':[0.01, 0.05, 0.1, 0.2, 0.3], 
     'gamma':[0, 0.1, 0.2, 0.5, 1.0], 
     'max_depth' : [6, 7, 8, 9, 1], 
    #  'subsample' : [0.6, 0.7, 0.8, 0.9, 1.0], 
    #  'colsample_bytree' : [0.6, 0.7, 0.8, 0.9, 1.0], 
     }

]

import xgboost as xgb
early_stop = xgb.callback.EarlyStopping(
    rounds=50,
    metric_name='mlogloss',
    data_name='validation_0',
    save_best=True
)

#2. 모델
model = HalvingRandomSearchCV(XGBClassifier(
                                          # tree_method='gpu_hist'
                                          tree_method='hist',
                                          device='cuda',
                                          n_estimators=500,
                                          eval_metric='mlogloss',
                                          callbacks=[early_stop]
                                         ),
                     parameters,
                     cv=kfold,
                     verbose=1, # 1:이터레이터 내용만 2이상은 훈련내용도 같이 출력
                     refit=True,
                    #  n_jobs=-1,
                    #  n_iter=10,
                     min_resources=30,
                     max_resources=1437,
                     random_state=4325,
                     factor=3,
                     aggressive_elimination=True,
                     )

#3. 훈련
start_time = time.time()
model.fit(x_train,y_train,
          eval_set=[(x_test, y_test)],
          verbose=True)
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

# 최적의 매개변수 : XGBClassifier(base_score=None, booster=None,
#               callbacks=[<xgboost.callback.EarlyStopping object at 0x0000020767C8E880>],
#               colsample_bylevel=None, colsample_bynode=None,
#               colsample_bytree=None, device='cuda', early_stopping_rounds=None,
#               enable_categorical=False, eval_metric='mlogloss',
#               feature_types=None, gamma=0, grow_policy=None,
#               importance_type=None, interaction_constraints=None,
#               learning_rate=0.1, max_bin=None, max_cat_threshold=None,
#               max_cat_to_onehot=None, max_delta_step=None, max_depth=9,
#               max_leaves=None, min_child_weight=None, missing=nan,
#               monotone_constraints=None, multi_strategy=None, n_estimators=500,
#               n_jobs=None, num_parallel_tree=None, objective='multi:softprob', ...)
# 최적의 파라미터 {'max_depth': 9, 'learning_rate': 0.1, 'gamma': 0}
# best_score :  0.9442527413541907
# model.score : 0.9638888888888889
# accuracy_score : 0.9638888888888889
# accuracy_score : 0.9638888888888889
# 걸린 시간 : 1761.03 초