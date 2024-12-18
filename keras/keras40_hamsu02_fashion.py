import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input
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


#2-1. 함수형 모델
input1 = Input(shape=(28,28,1))
conv2d1 = Conv2D(64,(4,4), strides=2, padding='same')(input1)
max = MaxPooling2D()(conv2d1)
conv2d2 = Conv2D(100, (2,2))(max)
conv2d3 = Conv2D(64, (2,2), activation='relu')(conv2d2)
flt1 =Flatten()(conv2d3)
dense1 = Dense(200, activation='relu')(flt1)
dense2 = Dense(100, activation='relu')(dense1)
dense3 = Dense(50, activation='relu')(dense2)
dense4 = Dense(25, activation='relu')(dense3)
dense5 = Dense(10, activation='softmax')(dense4)
model = Model(inputs=input1, outputs=dense5)


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

model.fit(x_train, y_train, epochs=100,  batch_size=3350,
                 verbose=2,
                 validation_split=0.3,
                 callbacks=[es]
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

