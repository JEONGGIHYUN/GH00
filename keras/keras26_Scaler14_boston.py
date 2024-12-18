#C:\TDS\ai5\study\keras\keras18_overfit1_boston.py

from tensorflow.keras.models import Sequential
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

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
scaler = RobustScaler()

scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)

print(x_train)
print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_train))



#2. 모델 구성
model = Sequential()
model.add(Dense(150, input_dim=13)) # 이미지 input_shape=(8,8,1)
# model.add(Dense(150, input_dim=13))
model.add(Dense(30))
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(7))
model.add(Dense(1))

#3. 컴파일 훈련

start_time = time.time()

model.compile(loss='mse', optimizer='adam')

hist = model.fit(x_train, y_train, epochs=50,  batch_size=32,
                 verbose=2,
                 validation_split=0.3)

end_time = time.time()

#4. 평가 예측

loss = model.evaluate(x_test, y_test)

results = model.predict(x_test)

print('로스 :', loss)

# print('보스턴 집값 :', results)

r2 = r2_score(y_test, results)

print('r2스코어 :', r2)

print('소요시간 :', round(end_time - start_time), '초')

# 로스 : 50.2800407409668
# 보스턴 집값 : [[30.493681 ]
# r2스코어 : 0.45516785479565147

# 로스 : 74.13037109375
# 보스턴 집값 : [[24.011402 ]
# r2스코어 : 0.19672669277870936

# MaxAbsScaler
# 로스 : 53.089210510253906
# r2스코어 : 0.42472787244370924
# 소요시간 : 1 초

# RobustScaler
# 로스 : 53.070220947265625
# r2스코어 : 0.42493341602406465
# 소요시간 : 1 초




