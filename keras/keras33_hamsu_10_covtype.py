import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

#1. 데이터
datasets = fetch_covtype()
# print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data # (581012, 54)
y = datasets.target # (581012,)
# print(x.shape, y.shape)

# print(np.unique(y, return_counts=True))
#  (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],
    #   dtype=int64))

# print(pd.value_counts(y))
# 2    283301
# 1    211840
# 3     35754
# 7     20510
# 6     17367
# 5      9493
# 4      2747
# dtype: int64

# y = pd.get_dummies(y)
# print(y)
# print(y.shape)

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape)

# from sklearn.preprocessing import OneHotEncoder
# y_ohe3 = y.reshape(-1, 1)
# y_ohe = OneHotEncoder(sparse=False) #True가 기본값
# y_ohe3 = y_ohe.fit_transform(y_ohe3)
# print(y_ohe3)
# print(y_ohe3.shape)


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1356, train_size=0.75, shuffle=True, stratify=y)

###############################################
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
scaler = RobustScaler()

scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)
################################################
# print(x_train)
print(np.min(x_train), np.max(x_train)) # 0.13 1680.0
print(np.min(x_test), np.max(x_test)) # 0.14 1480.0

# #2. 모델 구성
# model = Sequential ()
# model.add(Dense(500, input_dim=54, activation='relu'))
# model.add(Dense(250))
# model.add(Dense(125))
# model.add(Dropout(0.2))
# model.add(Dense(50))
# model.add(Dense(25))
# model.add(Dense(10))
# model.add(Dense(8, activation='softmax'))

#2. 함수형 모델 구성
input1 = Input(shape=(13, ))
dense1 = Dense(500, activation='relu')(input1)
dense2 = Dense(250)(dense1)
dense3 = Dense(125)(dense2)
dense4 = Dropout(0.2)(dense3)
dense5 = Dense(50)(dense4)
dense6 = Dense(25)(dense5)
dense7 = Dense(10)(dense6)
output1 = Dense(8, activation='softmax')(dense7)
model = Model(inputs=input1, outputs=output1)


#3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
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
filepath = ''.join([path, 'k32_10_', date, '_', filename])
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
model.fit(x_train, y_train, epochs=3000,  batch_size=10000,
                 verbose=2,
                 validation_split=0.3,
                 callbacks=[es, mcp]
                 )

#4. 예측 평가
loss = model.evaluate(x_test, y_test)
y_predict = np.around(model.predict(x_test))
accuracy_score = accuracy_score(y_test, y_predict)

print('loss :', loss)
print('acc score :', accuracy_score)

# loss : [0.4637472927570343, 0.8029851317405701]
# acc score : 0.7942693094118538
# time : 108.65 초

# loss : [1.3028264045715332, 0.6962679028511047]
# acc score : 0.6945674099674362
# time : 34.47 초

# loss : [0.6772934794425964, 0.7000131011009216]
# acc score : 0.6761788052570343
# time : 33.58 초

# MaxAbsScaler
# loss : [0.6784258484840393, 0.710587739944458]
# acc score : 0.6913660991511363
# time : 33.75 초

# loss : [0.684028685092926, 0.7165084481239319]
# acc score : 0.696364274748198
# time : 33.71 초

# RobustScaler
# loss : [1.1017342805862427, 0.6318974494934082]
# acc score : 0.6090270080480266
# time : 33.72 초

# loss : [2.7993953227996826, 0.5629487633705139]
# acc score : 0.5595615925316517
# time : 33.98 초

# ======================================================
# loss : [0.587283194065094, 0.7515645027160645]
# acc score : 0.737010595306121
