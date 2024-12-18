import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
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


print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape) # (10000, 32, 32, 3) (10000, 1)

x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)


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
model.add(Dense(170, input_shape=(32*32*3,)))
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

#3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

################
es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=10, verbose=1,
                   restore_best_weights=True)
################

model.fit(x_train, y_train, epochs=100,  batch_size=3350,
                 verbose=2,
                 validation_split=0.2,
                 callbacks=[es]
                 )

#4. 예측 평가
loss = model.evaluate(x_test, y_test)

y_predict = np.round(model.predict(x_test))

accuracy_score = accuracy_score(y_test, y_predict)

r2 = r2_score(y_test, y_predict)

print('r2_score : ', r2)
print('loss :', loss)
print('acc score :', accuracy_score)

# r2_score :  0.10444439384534117
# loss : [1.340731143951416, 0.5306000113487244]
# acc score : 0.3564

