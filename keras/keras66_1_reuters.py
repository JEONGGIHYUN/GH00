from tensorflow.keras.datasets import reuters
from keras.preprocessing.text import Tokenizer
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Conv1D, LSTM, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test,y_test) = reuters.load_data(
    num_words=1000,
    #maxlen=50,
    test_split=0.2    
)

print(x_train)
print(x_train.shape,x_test.shape) # (8982,) (2246,)
print(y_train.shape,y_test.shape) # (8982,) (2246,)
print(y_train)
print(np.unique(y_train))

print(len(np.unique(y_train)))

# print(type(x_train))
# print(len(x_train[0]), len(y_train[1]))
# print(len(x_train[0]))

print("뉴스기사의 최대길이 : ", max(len(i) for i in x_train)  )  
print("뉴스기사의 최소길이 : ", min(len(i) for i in x_train)  )  
print("뉴스기사의 평균길이 : ", sum(map(len, x_train)) /len(x_train)) 

#전처리
from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(x_train, padding='pre', maxlen=100,
                        truncating='pre')
x_test = pad_sequences(x_test, padding='pre', maxlen=100,
                        truncating='pre')

print(x_train.shape,x_test.shape) # (8982, 100) (2246, 100)
# print(y_train.shape,y_test.shape) # (8982,) (2246,)
# y 원핫 하고 만들기
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape,y_test.shape) # (8982, 46) (2246, 46)

# x_train = x_train.reshape(8982,25,4)
# y_train = y_train.reshape(8982,25,4)


#2 모델
model = Sequential()
model.add(Conv1D(filters=10, kernel_size=2, input_shape=(100,1)))
model.add(Flatten())
# model.add(LSTM(10))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(46, activation='softmax'))

#3 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train,y_train, epochs=100)

#4 예측 평가
loss = model.evaluate(x_test,y_test)
print('loss:', loss[0])

# y_predict = model.predict(x_test)

# r2 = r2_score(y_test, y_predict)

print('r2스코어 :', loss[1])


































































