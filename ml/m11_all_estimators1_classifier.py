import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
import warnings
import sklearn as sk
warnings.filterwarnings('ignore')
print('사이킷런 버전:',sk.__version__)
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

#2. 모델구성
all = all_estimators(type_filter='classifier')
# all = all_estimators(type_filter='regressor')

print('all : ', all)
print('모델의 갯수 : ', len(all))

print('사이킷런 버전:',sk.__version__)

for name, model in all:
    try:
        #2. 모델
        model = model()
        #3. 훈련
        model.fit(x_train,y_train)
        
        #4. 평가
        acc = model.score(x_test,y_test)
        print(name, '의 정답률 :', acc)
    except:
        print(name,'은 바보 다스베이더')