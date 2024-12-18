# https://www.kaggle.com/competitions/bike-sharing-demand/data

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input,Conv1D,Flatten,Reshape
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

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

# x.boxplot() # windspeed
# x.plot.box()
# plt.show()
x['windspeed'] = np.log1p(x['windspeed']) 

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.87, random_state=57543)

y_train = np.log1p(y_train)
y_test = np.log1p(y_test)

# 2 모델
model = RandomForestRegressor(random_state=1234,
                              max_depth=5,
                              min_samples_split=3,)

#3. 훈련
model.fit(x_train, y_train)



#4. 평가 예측
score = model.score(x_test, y_test)
print('score :', score)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2 score :', r2)

# 로그 변환전
# score : 0.3028950776275591
# r2 score : 0.3028950776275591

# 로그 변환 x만 한 후
# score : 0.30287249367634206
# r2 score : 0.30287249367634206

# 로그 변환  x,y둘 다
# score : 0.2811441959194322
# r2 score : 0.2811441959194322