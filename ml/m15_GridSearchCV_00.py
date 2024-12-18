import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score
import time


#1. 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=15426, shuffle=True, train_size=0.8, stratify=y)

n_splits = 5 
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=333)

parameters = [
    {'C':[1, 10, 100, 1000], 'kernel':['linear','sigmoid'], 'degree':[3,4,5]},
    {'C':[1, 10, 100], 'kernel':['rbf'], 'gamma':[0.001, 0.0001]},
    {'C':[1, 10, 100, 1000], 'kernel':['sigmoid'], 'gamma':[0.1, 0.01, 0.001, 0.0001], 'degree':[3,4]}
]

#2. 모델
model = GridSearchCV(SVC(), parameters, cv=kfold,
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

import pandas as pd
print(pd.DataFrame(model.cv_results_).T)

















































