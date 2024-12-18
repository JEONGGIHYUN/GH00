# https://dacon.io/competitions/official/235576/overview/description 대회 주소

import numpy as np
import pandas as pd # 인덱스와 칼럼을 분리하는데 사용하는 함수
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Reshape, Conv1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time

#1. 데이터
path = 'C:/TDS/ai5/_data/dacon/따릉이/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
print(train_csv) # [1459 rows x 10 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=0) 
print(test_csv) # [715 rows x 9 columns]

submission_csv = pd.read_csv(path + 'submission.csv', index_col=0) # [715 rows x 1 columns]
# print(submission_csv) # NaN으로 나오는 데이터는 없는 데이터를 뜻한다. 결측치 : 일반적인 데이터 집합에서 벗어난다는 뜻을 가진 이상치(outlier)의 하위 개념

# print(train_csv.shape) # (1459, 10)
# print(test_csv.shape) # (715, 9)
# print(submission_csv.shape) # (715, 1)

# print(train_csv.columns)
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
    #    'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
    #    'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
    #   dtype='object')

# print(train_csv.info())

############ 결측치 처리 1. 삭제 ##############
train_csv.isnull().sum()
# print(train_csv.isna().sum())

train_csv = train_csv.dropna() 
# print(train_csv.isna().sum())
# print(train_csv) # [1328 rows x 10 columns]

# print(test_csv.info())

test_csv = test_csv.fillna(test_csv.mean()) # 결측치 채우기 
# print(test_csv.info())

x = train_csv.drop(['count'], axis=1)
# print(x) # [1328 rows x 9 columns]

y = train_csv['count']
# print(y.shape) # (1328, )

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=4343)

################################################
from sklearn.preprocessing import MinMaxScaler , StandardScaler
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
model = Sequential()
model.add(Reshape(target_shape=(9,1)))
model.add(Conv1D(filters=10,kernel_size=2,input_shape=(9,1)))
model.add(Flatten())
model.add(Dense(500))
model.add(Dense(250))
model.add(Dense(125))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dense(1))

# #2. 함수형 모델 구성
# input1 = Input(shape=(9, ))
# dense1 = Dense(500)(input1)
# dense2 = Dense(250)(dense1)
# dense3 = Dense(125)(dense2)
# dense4 = Dropout(0.2)(dense3)
# dense5 = Dense(50)(dense4)
# output1 = Dense(1)(dense5)
# model = Model(inputs=input1, outputs=output1)


#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')

start_time = time.time()
################
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
################
es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=10, verbose=1,
                   restore_best_weights=True)
################
################파일 명 만 들 기
import datetime
date = datetime.datetime.now()
print(date)
print(type(date)) # <class 'datetime.datetime'>
date = date.strftime('%m%d_%H%M')
print(date)
print(type(date))

path = 'C:/TDS/ai5/study/_save/keras32_/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path, 'k32_04_', date, '_', filename])
################
################
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode = 'auto',
    verbose= 1,
    save_best_only=True,
    filepath = filepath
)
################
model.fit(x_train, y_train, epochs=3000,  batch_size=32,
                 verbose=2,
                 validation_split=0.3,
                 callbacks=[es, mcp]
                 )
end_time = time.time()

#4. 평가 예측
loss = model.evaluate(x_test, y_test)

results = model.predict(x_test)
# r2 = r2_score(y_test,  results)
# print('로스 :', loss)

y_submit = model.predict(test_csv)
# print(y_submit)
# print(y_submit.shape) # (715, 1)


# ######### submission.csv 만들기 // count 클럼에 값만 넣어 주면 된다. ####
submission_csv['count'] = y_submit
print(submission_csv)
print(submission_csv.shape)
r2 = r2_score(y_test,  results)
print('로스 :', loss)
print('소요시간 :', round(end_time - start_time), '초')


# submission_csv.to_csv(path + 'submission_0725_1746csv')



# 로스 : 2506.734375 train .87 random 434
# 로스 : 2447.269775390625 train .9 random 434
# 로스 : 2423.58203125 train .9 random 434
# 로스 : 2201.45849609375 train .9 random 4343434
# 로스 : 1739.7637939453125 train .9 random 4343
# 로스 : 1696.930419921875  train .9 random 4343 222 9 300 30 25 5 1 1000 
# 로스 : 1518.3536376953125 train .96 random 4343 222 9 300 30 25 5 1 1000 
# 로스 : 1488.0318603515625 train .98 random 4343 222 9 300 30 25 5 1 1000 
# 로스 : 1403.3687744140625 train .9875 random 4343 222 9 300 30 25 5 1 1000
# 로스 : 1368.3687744140625 train .9875 random 4343 222 9 300 30 25 5 1 1000
# 로스 : 1362.5675048828125 train .9822 random 4343 222 9 300 30 25 5 1 1000
# 로스 : 1361.5626220703125 train .9822 random 4343 222 9 300 30 25 5 1 1000
# 로스 : 1313.7088623046875 train .9822 random 4343 500 250 125 72.5 50 1 1000
# 로스 : 1282.2501220703125 train .9822 random 3535 500 250 125 72.5 50 1 1000
# 로스 : 1134.4781494140625 train .9822 random 5757 500 250 125 72.5 50 1 1000
# 로스 : 1016.1388549804688 train .9822 random 5757 500 250 125 72.5 50 1 1000
# 로스 : 1703.3221435546875 train .9 random 4343 222 9 300 30 25 5 1 1000 

# 로스 : 1694.4572

# loss: 1736.5221

# MaxAbsScaler
# loss: 1734.2111
# 로스 : 1824.1597900390625
# 로스 : 1729.641845703125

# RobustScaler
# 로스 : 1766.536376953125

# ----------------------------
# 로스 : 1807.3585205078125



# cpu 소요시간 : 3 초

# gpu 소요시간 : 4 초

# Conv1D
# 로스 : 1867.575439453125
# 소요시간 : 6 초