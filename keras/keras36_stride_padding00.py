import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# # print(x_train)
# # print(x_train[0])

# # print(y_train[0])


# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

# x_train = x_train.reshape(60000, 28, 28, 1)
# x_test = x_test.reshape(10000, 28, 28, 1)

# y_train = pd.get_dummies(y_train)
# y_test =  pd.get_dummies(y_test)

#2. 모델
model = Sequential()
model.add(Conv2D(10, (1,3), input_shape=(10, 10, 1),
                 strides=1,
                 padding='same')) #padding의 디폴트는 valid이고 same은 input_shape를 유지해 준다.
                       #shape = (batch_sitze, rows, culumes, channels)
model.add(Conv2D(filters=20, kernel_size=(3, 3),
                 strides=2,
                 )) # 25, 25, 20
                       #shape = (batch_size, heights, widths, channerls)
# model.add(Conv2D(15, (4, 4)))
# model.add(Flatten())
# model.add(Dense(units=150))
# model.add(Dense(units=100, input_shape=(150,)))
#                         # shape = (batch_size, input_dim)

# model.add(Dense(10, activation='softmax'))

model.summary()
'''
#3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=10,
                   restore_best_weights=True
                   )

model.fit(x_train, y_train, epochs=1, batch_size=3333,
          verbose=2,
          validation_split=0.2,
          callbacks=[es])

#4. 예측 평가
loss = model.evaluate(x_test, y_test)
y_predict = np.round(model.predict(x_test))
accuracy_score = accuracy_score(y_test, y_predict)


r2 = r2_score(y_test, y_predict)

print('r2_score : ', r2)
print('loss :', loss)
print('acc score :', accuracy_score)
'''