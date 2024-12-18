import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
a = np.array(range(1, 101))
size = 8
x_predict = np.array(range(96,106))   


def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)
    
bbb = split_x(a, size)
# print(bbb)
# print(bbb.shape)
x = bbb[:, :-1]
y = bbb[:, -1]
print(x, x.shape)
print(y, y.shape)
# x = x.reshape()
x_predict = split_x(x_predict, 6)


# #2 모델 구성
model = Sequential()
model.add(SimpleRNN(units=21, input_shape=(7,1), activation='relu')) # timesteps, features
model.add(Dense(500))
model.add(Dense(450))
model.add(Dense(400))
model.add(Dense(350))
model.add(Dense(300))
model.add(Dense(250))
model.add(Dense(200))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', )


es = EarlyStopping(monitor='loss', mode='min', 
                   patience=30, verbose=1,
                   restore_best_weights=True,
                   )

###### mcp 세이브 파일명 만들기 ######
# import datetime
# date = datetime.datetime.now()
# date = date.strftime("%m%d_%H%M")

# path = './_save/keras52/'
# filename = '{epoch:04d}-{loss:.4f}.hdf5' 
# filepath = "".join([save_csv, 'LSTM', date, '_', filename])   
#####################################

# mcp = ModelCheckpoint(
#     monitor='loss',
#     mode='auto',
#     verbose=1,     
#     save_best_only=True,   
#     filepath=filepath, 
# )

model.fit(x, y, epochs=1000,
        #   validation_split=0.1,
          callbacks=[es],
          verbose=1
          )

#4. 평가, 예측
result = model.evaluate(x, y)
print('loss :', result)

y_pred = model.predict(x_predict)
# y_pred = np.round(y_pred)
# y_pred = model.predict(x_predict.reshape(1,3,1))
print('결과 :', y_pred)    

