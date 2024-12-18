from tensorflow.keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Conv1D, LSTM, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

(x_train,y_train),(x_test,y_test) = imdb.load_data(num_words=1000)


# print(x_train)
# print(x_train.shape,x_test.shape) # (25000,) (25000,)
# print(y_train.shape,y_test.shape) # (25000,) (25000,)
# print(y_train)
# print(np.unique(y_train))

print(type(x_train)) # <class 'numpy.ndarray'>
print(type(y_train)) # <class 'numpy.ndarray'>

# print(type(x_train[0])) # <class 'list'>
# print(len(x_train[0]), len(y_train[1])) # 218 260

# print("최대길이 : ", max(len(i) for i in x_train)) # 2494
# print("최소길이 : ", min(len(i) for i in x_train)) # 11
# print("평균길이 : ", sum(map(len, x_train)) /len(x_train)) #238.71364

# x_train, x_test, y_train, y_test = train_test_split(x, xy_test, train_size=0.8, random_state=4)

# print(x_train.shape,x_test.shape) # (1, 25000) (1, 25000)
# print(np.shape(x_train),np.shape(x_test)) # (1, 25000) (1, 25000)

x_train = pad_sequences(x_train, padding='pre', maxlen=100,
                        truncating='pre')
x_test = pad_sequences(x_test, padding='pre', maxlen=100,
                       truncating='pre')

print(x_train.shape,x_test.shape) # (25000, 100) (25000, 100)

print(x_train, x_test)

#2 모델
#2 모델
model = Sequential()
model.add(Conv1D(filters=10, kernel_size=2, input_shape=(100,1)))
model.add(Flatten())
# model.add(LSTM(10))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3 컴파일 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train,y_train, epochs=100)

#4 예측 평가
loss = model.evaluate(x_test,y_test)
print('loss:', loss[0])

# y_predict = model.predict(x_test)

# r2 = r2_score(y_test, y_predict)

print('r2스코어 :', loss[1])



















































































































