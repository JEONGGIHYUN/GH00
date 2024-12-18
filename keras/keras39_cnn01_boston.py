from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Flatten
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

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=3)

print(x_train.shape, x_test.shape) # (430, 13) (76, 13)

x_train = x_train.reshape(430,13,1,1)
x_test = x_test.reshape(76,13,1,1)

x_train = x_train / 255.
x_test = x_test / 255.

#2. 모델 구성
model = Sequential()

model.add(Conv2D(64,kernel_size=(2,1), input_shape=(13,1,1))) 
model.add(Conv2D(50, (2,1), activation='relu'))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')

start_time = time.time()
################
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
################
es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=10, verbose=1,
                   restore_best_weights=True)
################

model.fit(x_train, y_train, epochs=50,  batch_size=32,
                 verbose=2,
                 validation_split=0.3,
                 callbacks=[es]
                 )
end_time= time.time()


#4. 평가 예측

loss = model.evaluate(x_test, y_test)

results = model.predict(x_test)

print('로스 :', loss)

r2 = r2_score(y_test, results)

print('r2스코어 :', r2)

print('소요시간 :', round(end_time - start_time), '초')

# cpu 소요시간 : 1 초

# gpu 소요시간 : 2 초