import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
              [9,8,7,6,5,4,3,2,1,0]])
y = np.array([1,2,3,4,5,6,7,8,9,10])

x = x.T

print(x.shape)
print(y.shape)

# #2. 모델 구성 (순차형)
# model = Sequential()
# model.add(Dense(10, input_shape=(3, )))
# model.add(Dense(9))
# model.add(Dense(8))
# model.add(Dense(7))
# model.add(Dense(6))
# model.add(Dense(5))
# model.add(Dense(1))
# model.summary()

#2-2 모델 구성 (함수형)
input1 = Input(shape=(3,))
dense1 = Dense(10, name='ys1')(input1)
dense2 = Dense(9, name='ys2')(dense1)
dense3 = Dense(8, name='ys3')(dense2)
dense4 = Dense(7, name='ys4')(dense3)
output1 = Dense(1)(dense4)
model = Model(inputs=input1, outputs=output1)
model.summary()