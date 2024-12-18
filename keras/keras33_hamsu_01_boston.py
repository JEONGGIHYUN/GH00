from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input
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


# #2. 모델 구성
# model = Sequential()
# model.add(Dense(64, input_dim=13)) # 이미지 input_shape=(8,8,1)
# model.add(Dropout(0.3))
# model.add(Dense(64))
# model.add(Dense(32))
# model.add(Dense(32))
# model.add(Dense(16))
# model.add(Dense(8))
# model.add(Dense(1))

#2. 함수형 모델 구성
input1 = Input(shape=(13, ))
dense1 = Dense(64)(input1)
dense2 = Dropout(0.3)(dense1)
dense3 = Dense(64)(dense2)
dense4 = Dense(32)(dense3)
dense5 = Dense(32)(dense4)
dense6 = Dense(16)(dense5)
dense7 = Dense(8)(dense6)
output1 = Dense(1)(dense7)
model = Model(inputs=input1, outputs=output1)

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')

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
filepath = ''.join([path, 'k32_01_', date, '_', filename])
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
model.fit(x_train, y_train, epochs=50,  batch_size=32,
                 verbose=2,
                 validation_split=0.3,
                 callbacks=[es, mcp]
                 )


# model.save('C:/TDS/ai5/study/_save/keras29/keras29_mcp3.h5')

#4. 평가 예측

loss = model.evaluate(x_test, y_test)

results = model.predict(x_test)

print('로스 :', loss)

r2 = r2_score(y_test, results)

print('r2스코어 :', r2)

