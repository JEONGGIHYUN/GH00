import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
import time
from sklearn.model_selection import train_test_split

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


# print(x_train)
# print(x_train[0])

# print(y_train[0])


print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28 * 28)
x_test = x_test.reshape(10000, 28 * 28)



#######스케일링
x_train = x_train / 255.
x_test = x_test / 255.
print(np.max(x_train),np.min(x_train))

########스케일링 1-2
# x_train=(x_train - 127.5) / 127.5
# x_test=(x_test - 127.5) / 127.5

########스케일링 2. MinMaxScaler
# scaler = MinMaxScaler()

### 원핫 케라스
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# print(np.max(x_train), np.min(x_train))


### 원핫 판다스
# y_train = pd.get_dummies(y_train)
# y_test =  pd.get_dummies(y_test)

print(y_train.shape, y_test.shape)

#2. 모델
model = Sequential()
model.add(Dense(units=200, activation='relu'))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=50))
model.add(Dense(units=25, activation='relu'))
                        # shape = (batch_size, input_dim)
model.add(Dense(10, activation='softmax'))

# model.summary()
#3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

start_time = time.time()

es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=10, verbose=1,
                   restore_best_weights=True)
################

model.fit(x_train, y_train, epochs=100,  batch_size=3350,
                 verbose=2,
                 validation_split=0.3,
                 callbacks=[es]
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