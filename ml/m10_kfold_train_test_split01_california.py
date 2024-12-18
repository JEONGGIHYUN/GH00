from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. 데이터
dataset = fetch_california_housing()

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=3, shuffle=True)
################################################
from sklearn.preprocessing import MinMaxScaler , StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
scaler = RobustScaler()
import numpy as np

scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)
################################################
################################################
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=False,)
            #   random_state=333)

# kfold 할 준비 끗



#2. 모델
model = SVR()

#4. 평가 
scores = cross_val_score(model, x_train, y_train, cv=kfold) # 기준 점수 학인
print('ACC : ', scores, '\n 평균 ACC : ', round(np.mean(scores), 4))

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
print(y_predict)
print(y_test)

acc = accuracy_score(y_test, y_predict)
print('cross_val_predict ACC :', acc)
###############################################
# print(x)
# print(y)
# print(x.shape, y.shape) #(20640, 8) (20640, )

# #2. 모델 구성
# model = Sequential()
# model.add(Dense(341, input_dim=8))
# model.add(Dense(125))
# model.add(Dropout(0.2))
# model.add(Dense(25))
# model.add(Dense(15))
# model.add(Dense(1))

# #2. 함수형 모델 구성
# input1 = Input(shape=(8, ))
# dense1 = Dense(341)(input1)
# dense2 = Dense(125)(dense1)
# dense3 = Dropout(0.2)(dense2)
# dense4 = Dense(25)(dense3)
# dense5 = Dense(15)(dense4)
# output1 = Dense(1)(dense5)
# model = Model(inputs=input1, outputs=output1)


# #3. 컴파일 훈련
# model.compile(loss='mse', optimizer='adam')
# ################
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# ################
# es = EarlyStopping(monitor='val_loss', mode='min',
#                    patience=10, verbose=1,
#                    restore_best_weights=True)
# ################
# ################파일 명 만 들 기
# import datetime
# date = datetime.datetime.now()
# print(date)
# print(type(date)) # <class 'datetime.datetime'>
# date = date.strftime('%m%d_%H%M')
# print(date)
# print(type(date))

# path = 'C:/TDS/ai5/study/_save/keras32_/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
# filepath = ''.join([path, 'k32_02_', date, '_', filename])
# ################
# ################
# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode = 'auto',
#     verbose= 1,
#     save_best_only=True,
#     filepath = filepath
# )
# ################
# model.fit(x_train, y_train, epochs=3000,  batch_size=32,
#                  verbose=2,
#                  validation_split=0.3,
#                  callbacks=[es, mcp]
#                  )

# #4. 평가 예측
# loss = model.evaluate(x_test, y_test, verbose=0)
# results = model.predict(x_test)
# r2 = r2_score(y_test, results)
# print('로스 :', loss)
# print('r2스코어 :', r2)