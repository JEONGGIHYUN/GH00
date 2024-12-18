import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPooling2D
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
import time
from sklearn.model_selection import train_test_split


#1. 데이터

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape) # (10000, 32, 32, 3) (10000, 1)

x_train = x_train.reshape(50000,32*32*3)
x_test = x_test.reshape(10000,32*32*3)

#1-1 스케일링
x_train = x_train / 255.
x_test = x_test / 255.

#1-2 원핫 인코딩
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



print(y_train.shape, y_test.shape) # (50000, 100) (10000, 100)

#2. 모델
model = Sequential()
model.add(Dense(600, input_shape=(32*32*3,)))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(500, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(100, activation='softmax'))

#3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=10, verbose=1,
                   restore_best_weights=True)


model.fit(x_train, y_train, epochs=100,  batch_size=546,
                 verbose=2,
                 validation_split=0.2,
                 callbacks=[es]
                 )

#4. 예측 평가
loss = model.evaluate(x_test, y_test)
y_pred = np.round(model.predict(x_test))

accuracy_score = accuracy_score(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

print('r2_score : ', r2)
print('loss :', loss)
print('acc score :', accuracy_score)
