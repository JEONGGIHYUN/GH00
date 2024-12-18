# https://www.kaggle.com/datasets/maciejgronczynski/biggest-genderface-recognition-dataset/data

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time


np_path = 'C:/TDS/ai5/_data/_save_npy/'

x_train = np.load(np_path + 'keras45_07_x_train_gender.npy')

y_train = np.load(np_path + 'keras45_07_y_train_gender.npy')

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.8, random_state=4)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(60,(4,4), input_shape=(100,100,3)))
model.add(Conv2D(60, (2,2), activation='relu', strides=2))
model.add(Conv2D(60, (4,4), activation='relu'))
model.add(Conv2D(60, (4,4), activation='relu'))
model.add(Conv2D(60, (4,4), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(60, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) # accuracy, mse

start_time = time.time()
################
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
################
################
es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=10, verbose=1,
                   restore_best_weights=True)
################
import datetime

date = datetime.datetime.now()

date = date.strftime('%m%d_%H%M')

filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([np_path, 'k45_가중치_', date,'_', filename])

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode = 'auto',
    verbose= 1,
    save_best_only=True,
    filepath = filepath
)

model.fit(x_train, y_train, epochs=30,  batch_size=100,
                 verbose=1,
                 validation_split=0.3,
                 callbacks=[es,mcp]
                 )

end_time = time.time()

#4. 평가 예측
loss = model.evaluate(x_test, y_test)


y_pred = model.predict(x_test)
print(y_pred)

y_pred = np.round(y_pred)

accuracy_score = accuracy_score(y_test, y_pred)


print('로스 :', loss)
print('acc_score :', accuracy_score)
print('소요시간 :', round(end_time - start_time), '초')


