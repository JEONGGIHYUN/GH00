import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

save_csv = 'C:/TDS/ai5/_data/LSTM_scale save/'

#1. 데이터

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60]              
              ])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50,60,70])        # 80 맞추기

x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)      # (13, 3, 1)


#2 모델구성
model = Sequential()
model.add(SimpleRNN(units=21, input_shape=(3,1), activation='relu')) # timesteps, features
model.add(Dense(20)) # RNN은 Dense와 바로 연결이 가능하다.
model.add(Dense(400, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(75))
model.add(Dense(50))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='loss', mode='min', 
                   patience=30, verbose=1,
                   restore_best_weights=True,
                   )

###### mcp 세이브 파일명 만들기 ######
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

path = './_save/keras52/'
filename = '{epoch:04d}-{loss:.4f}.hdf5' 
filepath = "".join([save_csv, 'LSTM', date, '_', filename])   
#####################################

mcp = ModelCheckpoint(
    monitor='loss',
    mode='auto',
    verbose=1,     
    save_best_only=True,   
    filepath=filepath, 
)

model.fit(x, y, epochs=1000,
        #   validation_split=0.1,
          callbacks=[es,mcp],
          verbose=1
          )

#4. 평가, 예측
result = model.evaluate(x, y)
print('loss :', result)


y_pred = model.predict(x_predict.reshape(1,3,1))
print('[50,60,70]의 결과 :', y_pred)    # 80 나오기

# loss : 1.2464021892810706e-05
# [50,60,70]의 결과 : [[80.00153]]

# loss : 1.601638359716162e-06
# [50,60,70]의 결과 : [[79.98797]]

# loss : 1.7931261027115397e-05
# [50,60,70]의 결과 : [[80.0081]]