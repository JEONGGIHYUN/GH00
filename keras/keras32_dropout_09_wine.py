from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import time

#1. 데이터
datasets = load_wine()
# print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data # (178, 13)
y = datasets.target # (178,)
# print(x.shape, y.shape)

# print(np.unique(y, return_counts=True))
# (array([0, 1, 2]), array([59, 71, 48], dtype=int64))

# print(pd.value_counts(y))
# (array([0, 1, 2]), array([59, 71, 48], dtype=int64))
# 1    71
# 0    59
# 2    48

y = pd.get_dummies(y)
print(y)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=135, train_size=0.7, shuffle=True, stratify=y)

###############################################
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
scaler = RobustScaler()

scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)
################################################
# # print(x_train)
# print(np.min(x_train), np.max(x_train)) # 0.13 1680.0
# print(np.min(x_test), np.max(x_test)) # 0.14 1480.0

#2. 모델 구성
model = Sequential()
model.add(Dense(500, input_dim=13, activation='relu'))
model.add(Dense(231, activation='relu'))
model.add(Dense(131, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))

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
filepath = ''.join([path, 'k32_09_', date, '_', filename])
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
model.fit(x_train, y_train, epochs=3000,  batch_size=200,
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

# loss : [0.1156327873468399, 0.9444444179534912]
# time : 12.22 초
# acc score : 0.9444444444444444

# loss : [0.18587186932563782, 0.9444444179534912]
# time : 12.3 초
# acc score : 0.94444444444

# loss : [0.44451165199279785, 0.8703703880310059]
# time : 12.3 초
# acc score : 0.8703703703703703

# MaxAbsScaler
# loss : [1.0893439054489136, 0.3888888955116272]
# time : 12.59 초
# acc score : 0.0

# loss : [0.21064040064811707, 0.9074074029922485]
# time : 12.42 초
# acc score : 0.8703703703703703

# RobustScaler
# loss : [0.2191983163356781, 0.8888888955116272]
# time : 12.32 초
# acc score : 0.8888888888888888

# loss : [0.2118058055639267, 0.9629629850387573]
# time : 12.01 초
# acc score : 0.9444444444444444

# -----------------------------------------------------
# loss : [0.7284064292907715, 0.6851851940155029]
# acc score : 0.6111111111111112