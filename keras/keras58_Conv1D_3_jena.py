# https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016/code
# https://www.kaggle.com/code/tila123/starter-jena-climate-2009-2016-627dd05e-6/notebook

'''
[실습] y는 T (degC), 자를는거는 마음대로
31.12.2016 00:10:00 부터 01.01.2017 00:00:00 까지는 사용 X
x shape = (n, 720(5일*144), 13), y shape = (n, 144), predict = (1, 144)
평가지표 : RMSE
'''

import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Dropout, BatchNormalization,Flatten, Bidirectional, Conv1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.utils import to_categorical
import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async" 

#1. 데이터
path = "C:/TDS/ai5/_data/kaggle/jena/"
csv = pd.read_csv(path + "jena_climate_2009_2016.csv", index_col=0)
# print(csv.shape)            # (420551, 14)

train_dt = pd.DatetimeIndex(csv.index)

csv['day'] = train_dt.day
csv['month'] = train_dt.month
csv['year'] = train_dt.year
csv['hour'] = train_dt.hour
csv['dos'] = train_dt.dayofweek

y3= csv.tail(144)
y3 = y3['T (degC)']

csv = csv[:-144]
# print(csv)
# print(csv.shape)          # (420407, 19)

x = csv.drop(['T (degC)',"wv (m/s)","max. wv (m/s)","wd (deg)","year"], axis=1)
y = csv['T (degC)']
# print(x.shape, y.shape)   # (420407, 14) (420407,)

size = 144

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

x1 = split_x(x, size)
# print(x)
# print(x.shape)            # (420264, 144, 14)

y = split_x(y, size)
# print(x)
# print(y.shape)            # (420264, 144)
x = x1[:-1]
y = y[1:]

x_test2 = x1[-1]
x_test2 = np.array(x_test2).reshape(1, 144, 14)

# print(x.shape)            # (420263, 144, 10)
# print(y.shape)            # (420263, 144)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=4)
# print(x_train.shape, x_test.shape)  # (336210, 144, 14) (84053, 144, 14)
# print(y_train.shape, y_test.shape)  # (336210, 144) (84053, 144)

# #2. 모델구성
model = Sequential()
model.add(Conv1D(filters=(64),kernel_size=(60), input_shape=(144,14)))
model.add(Conv1D(filters=(32),kernel_size=(20)))
model.add(Flatten())
# model.add(GRU(128))
model.add(Dense(300))
model.add(Dropout(0.2))
model.add(Dense(280))
model.add(Dropout(0.2))
model.add(Dense(260))
model.add(Dense(240))
model.add(Dense(220))
model.add(Dense(144))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='val_loss', mode='min', patience=30, verbose=1, restore_best_weights=True,)

########## mcp 세이브 파일명 만들기 시작 ##########
# import datetime
# date = datetime.datetime.now()
# print(date)
# print(type(date))

# date = date.strftime("%m%d.%H%M")
# print(date)
# print(type(date))

# path = './_save/keras55/'
# filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5'
# filepath = "".join([path, 'k55_jena_date_', date, '_epo_', filename])

########## mcp 세이브 파일명 만들기 끝 ##########

# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

start = time.time()
model.fit(x_train, y_train, epochs=200, batch_size=1024, validation_split=0.2,callbacks=[es])
end = time.time()

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print("loss : ", results)

y_predict = model.predict(x_test2)
# print(y_predict.shape)  # (1, 144)

# y_predict = np.array(y_predict).reshape(144, 1)
y_predict = y_predict.T
# print(y_predict.shape)  # (144, 1)

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y3, y_predict)
print('RMSE : ', rmse)

print("time :", round(end-start,2),'초')

# submit_csv['T (degC)'] = y_submit
# print(submit_csv)                  # [6493 rows x 1 columns]
# print(submit_csv.shape)            # (6493, 1)

# submit_csv.to_csv(path + "sampleSubmission_0809.csv")

# SimpleRNN + LSTM
# loss :  10.113434791564941
# RMSE :  3.0443296236064548
# time : 582.43 초


