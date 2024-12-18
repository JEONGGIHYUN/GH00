import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits
from sklearn.model_selection import train_test_split,StratifiedKFold, KFold, cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import all_estimators
import sklearn as sk
import warnings
import time
warnings.filterwarnings('ignore')

#cross val score - 교차검증점수
# 5개로 짜른 데이터들마다 교차 검증 점수를 매긴다

#1. 데이터
iris = load_iris(return_X_y=True)
cancer = load_breast_cancer(return_X_y=True)
wine = load_wine(return_X_y=True)
digits = load_digits(return_X_y=True)

all = all_estimators(type_filter='classifier')


datasets = [iris, cancer, wine, digits]
data_name = ['아이리스', '캔서', '와인', '디저트']

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=787)

start_time = time.time()

for index, value in enumerate(datasets):
    x, y = value
    x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, train_size=0.8,
    stratify=y,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    for name, model in all:
        try:
            #2. 모델
            model = model()
            #3. 훈련
            scores = cross_val_score(model, x_train, y_train , cv= kfold)
            print('=========================',data_name[index],name,'========================')
            print('ACC : ', scores, '\n 평균 ACC : ', round(np.mean(scores), 4))
        
            y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
            acc = accuracy_score(y_test,y_predict)
            print('cross_val_predict ACC :', acc)
        
            #4. 평가
            acc = model.score(x_test,y_test)
            print(name, '의 정답률 :', acc)
        except:
            print(name,'은 바보 다스베이더')

end_time = time.time()
print('걸린 시간 :', round(end_time - start_time, 2), '초')