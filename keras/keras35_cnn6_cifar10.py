import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
import time
from sklearn.model_selection import train_test_split

#1. 데이터

(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# print(x_train)
# print(x_train[0])

# print(y_train[0])


# print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape) # (10000, 32, 32, 3) (10000, 1)

#1-1 스케일링
x_train = x_train / 255.
x_test = x_test / 255.
# print(np.max(x_train),np.min(x_train)

#1-2 원핫 인코딩
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape, y_test.shape)

# 2. 모델
model = Sequential()
model.add(Conv2D(32, (4, 4), input_shape=(32, 32, 3)))
model.add(Conv2D(100, (2, 2), activation='relu'))
model.add(Conv2D(50, (2, 2), activation='relu'))
model.add(Conv2D(30, (2, 2), activation='relu'))
model.add(Flatten())
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

start_time = time.time()

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

path = 'C:/TDS/ai5/study/_save/keras35_/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path, 'k35_06_', date, '_', filename])
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
model.fit(x_train, y_train, epochs=100,  batch_size=3350,
                 verbose=2,
                 validation_split=0.2,
                 callbacks=[es, mcp]
                 )

end_time = time.time()

#4. 예측 평가
loss = model.evaluate(x_test, y_test)

y_predict = np.round(model.predict(x_test))

accuracy_score = accuracy_score(y_test, y_predict)

r2 = r2_score(y_test, y_predict)

print('r2_score : ', r2)
print('loss :', loss)
print('acc score :', accuracy_score)



