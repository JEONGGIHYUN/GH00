# https://www.kaggle.com/competitions/bike-sharing-demand/data

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. 데이터
# path = './_data/bike_sharing_demand/' # 상대 경로 방식

# * 3개 다 같은 방식이다 *
# path = 'C:\\TDS\\ai5\\_data\\bike-sharing-demand' # 절대 경로 방식
# path = 'C://TDS//ai5//_data//bike-sharing-demand//' 
path = 'C:/TDS/ai5/_data/bike-sharing-demand/'

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

# print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.87, random_state=57543)

################################################
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
scaler = RobustScaler()

scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)
################################################
# print(x_train)
# print(np.min(x_train), np.max(x_train)) # 
# print(np.min(x_test), np.max(x_test)) # 

#2. 모델 구성

#3. 컴파일 훈련

#4. 평가 예측

print('================== 1. save.model 출력====================')
model = load_model('C:/TDS/ai5/study/_save/keras30_mcp/k30_05_0726_1943_0090-23077.8008.hdf5')

loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)

# print(results)


# y_submit = model.predict(test_csv)
# # print(y_submit)
# # print(y_submit.shape)

# sampleSubmission['count'] = y_submit
# print(sampleSubmission)
# print(sampleSubmission.shape)
r2 = r2_score(y_test, results)
print('로스 :', loss)
print('r2 :', r2)
# sampleSubmission.to_csv(path + 'submission_0717_1748.csv')

# 로스 : 23211.236328125
# r2 : 0.22942162147049994

# 로스 : 21758.42578125
# r2 : 0.31953088138753194

# 로스 : 23586.23828125
# r2 : 0.3230813282133992

# 로스 : 22588.06640625
# r2 : 0.35172867666926644

# 로스 : 22253.794921875
# r2 : 0.32804621675522283 random 5790

# 로스 : 21562.87890625
# r2 : 0.3212597537890469

# 로스 : 20815.568359375
# r2 : 0.3285026919423433

# 로스 : 23061.962890625
# r2 : 0.2759931261375891

# 로스 : 22538.07421875
# r2 : 0.29244005653203875

# MaxAbsScaler
# 로스 : 22928.064453125
# r2 : 0.2801967342598858

# 로스 : 22757.578125
# r2 : 0.2855489816594161

# RobustScaler
# 로스 : 23112.58203125
# r2 : 0.27440386813691053

# 로스 : 23056.5234375
# r2 : 0.27616387076948223

