# https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016/code

import numpy as np
import pandas as pd
import time
import datetime
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense,SimpleRNN,LSTM,GRU,Dropout,Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.utils import to_categorical
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async" 

#1. 데이터
path = "./_save/keras55/"
csv = pd.read_csv(path + "jena_climate_2009_2016.csv", index_col=0)
save_csv = pd.read_csv(path + "jena_climate_2009_2016.csv", index_col=0)

train_dt = pd.DatetimeIndex(csv.index)
csv['day'] = train_dt.day
csv['month'] = train_dt.month
csv['year'] = train_dt.year
csv['hour'] = train_dt.hour
csv['dos'] = train_dt.dayofweek

test = csv.tail(144)
test = test['T (degC)']

csv = csv[:-144]
x = csv.drop(['T (degC)',"wv (m/s)","max. wv (m/s)","wd (deg)","year"], axis=1)
y = csv['T (degC)']

size = 144

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

x = split_x(x, size)
y = split_x(y, size)

x = x[:-1]
y = y[1:]
 
pred = x[-1]
pred = np.array(pred).reshape(1, 144, 14)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=4)

'''
#2. 모델구성
model = Sequential()
model.add(GRU(16, input_shape=(144,14),return_sequences=True))
model.add(GRU(32))
model.add(Dense(400))
model.add(Dropout(0.2))
model.add(Dense(370))
model.add(Dropout(0.2))
model.add(Dense(320))
model.add(Dense(300))
model.add(Dense(288))
model.add(Dense(144))

#3 컴파일 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1, restore_best_weights=True,)

########## mcp 세이브 파일명 만들기 시작 ##########
date = datetime.datetime.now()
# print(date)
# print(type(date))

date = date.strftime("%m%d.%H%M")
# print(date)
# print(type(date))

path = './_save/keras55/'
filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5'
filepath = "".join([path, 'k55_jena_date_', date, '_epo_', filename])

########## mcp 세이브 파일명 만들기 끝 ##########

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=1024, validation_split=0.2, callbacks=[es, mcp])
end = time.time()
'''
#4. 평가, 예측

print('=============모델 출력==============')
model = load_model('./_save/keras55/jena_정기현.hdf5')


loss = model.evaluate(x_test, y_test)

y_predict = model.predict(pred)

y_predict = y_predict.T

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(test, y_predict)

print("loss : ", loss)

print('RMSE : ', rmse)

# print("time :", round(end-start,2),'초')

submit = pd.read_csv(path + "jena_climate_2009_2016.csv")

submit = submit[['Date Time','T (degC)']]
submit = submit.tail(144)
# print(submit)

# y_submit = pd.DataFrame(y_predict)
# print(y_submit)

submit['T (degC)'] = y_predict
# print(submit)                  # [6493 rows x 1 columns]
# print(submit.shape)            # (6493, 1)

submit.to_csv(path + "jena_정기현.csv", index=False)























































































