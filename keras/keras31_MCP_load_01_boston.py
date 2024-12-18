from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston

#1. 데이터
dataset = load_boston()

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=1,shuffle=True)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)

#2. 모델 구성

#3. 컴파일 훈련
################
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
################

#4. 평가 예측

print('================== 1. save.model 출력====================')
model = load_model('C:/TDS/ai5/study/_save/keras30_mcp/k30_01_0726_1940_0141-36.7957.hdf5')

loss = model.evaluate(x_test, y_test)

results = model.predict(x_test)



print('로스 :', loss)

# print('보스턴 집값 :', results)

r2 = r2_score(y_test, results)

print('r2스코어 :', r2)