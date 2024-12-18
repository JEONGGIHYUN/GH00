import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
#cross val score - 교차검증점수
# 5개로 짜른 데이터들마다 교차 검증 점수를 매긴다

#1. 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, train_size=0.8,
    stratify=y,
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# df = pd.DataFrame(datasets.data, columns = datasets.feature_names)
# print(df)

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=False,)
            #   random_state=333)

# kfold 할 준비 끗



#2. 모델
model = SVC()

#4. 평가 
scores = cross_val_score(model, x_train, y_train, cv=kfold) # 기준 점수 학인
print('ACC : ', scores, '\n 평균 ACC : ', round(np.mean(scores), 4))

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
print(y_predict)
print(y_test)

acc = accuracy_score(y_test, y_predict)
print('cross_val_predict ACC :', acc)