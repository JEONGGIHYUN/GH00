import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
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

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)



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
model.add(Conv2D(64, (4,4), input_shape=(28, 28, 1),
                 strides=2,
                 padding='same')) # 27, 27 , 10
                       #shape = (batch_sitze, rows, culumes, channels)
model.add(Conv2D(filters=100, kernel_size=(2, 2))) # 25, 25, 20
                       #shape = (batch_size, heights, widths, channerls)
model.add(Conv2D(64, (2, 2), activation='relu')) #24, 24, 32
model.add(Flatten())
model.add(Dense(units=200, activation='relu'))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=50, input_shape=(32,)))
model.add(Dense(units=25, activation='relu'))
                        # shape = (batch_size, input_dim)
model.add(Dense(10, activation='softmax'))

# model.summary()
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
filepath = ''.join([path, 'k35_05_', date, '_', filename])
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
                 validation_split=0.3,
                 callbacks=[es, mcp]
                 )

end_time = time.time()

#4. 예측 평가
loss = model.evaluate(x_test, y_test)

y_predict = np.round(model.predict(x_test, 8))

accuracy_score = accuracy_score(y_test, y_predict)

r2 = r2_score(y_test, y_predict)

print('r2_score : ', r2)
print('loss :', loss)
print('acc score :', accuracy_score)