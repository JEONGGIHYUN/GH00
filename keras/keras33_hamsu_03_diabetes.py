from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np


#1. 데이터
datasets = load_diabetes()

x = datasets.data
y = datasets.target

# print(x.shape, y.shape) (442, 10) (442, )

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=7251)
################################################
from sklearn.preprocessing import MinMaxScaler , StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
scaler = RobustScaler()

scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)
################################################
# print(x_train)
# print(np.min(x_train), np.max(x_train)) # -0.137767225690012 0.198787989657293
# print(np.min(x_test), np.max(x_test)) # -0.112794729823292 0.145012221505454

# #2. 모델 구성
# model = Sequential()
# model.add(Dense(251, input_dim=10))
# model.add(Dense(141))
# model.add(Dense(171))
# model.add(Dropout(0.2))
# model.add(Dense(5))
# model.add(Dense(1))

#2. 함수형 모델 구성
input1 = Input(shape=(10, ))
dense1 = Dense(251)(input1)
dense2 = Dense(141)(dense1)
dense3 = Dense(171)(dense2)
dense4 = Dropout(0.2)(dense3)
dense5 = Dense(5)(dense4)
output1 = Dense(1)(dense5)
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
filepath = ''.join([path, 'k32_03_', date, '_', filename])
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

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
r2 = r2_score(y_test, results)
print('로스 :', loss)
print('r2스코어 :', r2)

# 로스 : 2607.511962890625
# r2스코어 : 0.5607322201160404 train .9 random 5 epochs 1000

# 로스 : 2431.83935546875
# r2스코어 : 0.5818174376445646 train .9 random 9 epochs 1000

# 로스 : 2407.46728515625
# r2스코어 : 0.5860084929864275 train .9 random 9 epochs 1000

# 로스 : 2405.62158203125
# r2스코어 : 0.5863258357446803  train .9 random 9 epochs 1000

# 로스 : 2832.931884765625
# r2스코어 : 0.6184719992107199 train .9 random 52151 epochs 1000

# 로스 : 2817.88916015625
# r2스코어 : 0.6204978866983641 train .9 random 52151 epochs 1000

# 로스 : 2803.8544921875
# r2스코어 : 0.6223879836278867 train .9 random 52151 epochs 1000

# 로스 : 2120.6298828125 
# r2스코어 : 0.6278116967478458 train .9 random 7251 epochs 1000

# 로스 : 2056.626708984375
# r2스코어 : 0.6390448036614433 train .9 random 7251 epochs 1000

# 로스 : 2039.3199462890625
# r2스코어 : 0.6420822125482439 train .9 random 7251 epochs 1000

# 로스 : 2046.4427490234375
# r2스코어 : 0.6408321012210374

# 로스 : 2053.68505859375
# r2스코어 : 0.6395610748991116

# MaxAbsScaler
# 로스 : 2032.602294921875
# r2스코어 : 0.643261240131939

# 로스 : 2052.7490234375
# r2스코어 : 0.6397253203110789

# RobustScaler
# 로스 : 2071.18212890625
# r2스코어 : 0.6364901717549691

# 로스 : 2051.2138671875
# r2스코어 : 0.6399947242105355

# -----------------------------------------
# 로스 : 2362.23779296875
# r2스코어 : 0.5854073896980578

