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

# #2. 모델 구성
# model = Sequential()
# model.add(Dense(32, input_shape=(13,))) # 이미지 input_shape=(8,8,1)
# model.add(Dense(32))
# model.add(Dense(16))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1))

#3. 컴파일 훈련

# start_time = time.time()

# model.compile(loss='mse', optimizer='adam')

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# es = EarlyStopping(monitor='val_loss', mode='min',
#                    patience=10, verbose=1,
#                    restore_best_weights=True)

# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode = 'auto',
#     verbose= 1,
#     save_best_only=True,
#     filepath='C:/TDS/ai5/study/_save/keras29/keras29_mcp1.hdf5'
# )

# hist = model.fit(x_train, y_train, epochs=3000,  batch_size=32,
#                  verbose=2,
#                  validation_split=0.3,
#                  callbacks=[es, mcp]
#                  )

# end_time = time.time()

model = load_model('C:/TDS/ai5/study/_save/keras29/keras29_mcp1.hdf5')

#4. 평가 예측

loss = model.evaluate(x_test, y_test)

results = model.predict(x_test)

print('로스 :', loss)

# print('보스턴 집값 :', results)

r2 = r2_score(y_test, results)

print('r2스코어 :', r2)

# print('소요시간 :', round(end_time - start_time), '초')
