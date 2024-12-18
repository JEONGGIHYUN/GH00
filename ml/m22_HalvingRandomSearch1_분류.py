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

#2. 모델
model = HalvingRandomSearchCV(XGBClassifier(
                                          # tree_method='gpu_hist'
                                          tree_method='hist',
                                          device='cuda',
                                          n_estimators=500,
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
          verbose=False)
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