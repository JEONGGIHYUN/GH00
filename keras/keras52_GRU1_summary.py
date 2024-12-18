import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM,GRU #기초가 되는 RNN(Recurrent Neural Network)이다
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
dataset = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3,],
              [2,3,4,],
              [3,4,5,],
              [4,5,6,],
              [5,6,7,],
              [6,7,8,],
              [7,8,9,],]
             )

y = np.array([4,5,6,7,8,9,10,])

print(x.shape, y.shape) # (7, 3) (7,)

x = x.reshape(x.shape[0],x.shape[1], 1)

# print(x.shape) (7, 3, 1)
# 3-D tensor with shape (batch_size, timesteps, features*)

#2 모델구성
model = Sequential()
# model.add(SimpleRNN(units=21, input_shape=(3,1), activation='relu'))
# model.add(LSTM(units=10, input_shape=(3,1), activation='relu'))
model.add(GRU(units=10, input_shape=(3,1), activation='relu')) # timesteps, features
model.add(Dense(10)) # RNN은 Dense와 바로 연결이 가능하다.
model.add(Dense(15, activation='relu'))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

# Model: "sequential"  3 * (다음 노드 수^2 +  다음 노드 수 * Shape 의 feature + 다음 노드수 ) 
# 3 * (10*10 + 10 * 1 + 10)
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  gru (GRU)                   (None, 10)                390

#  dense (Dense)               (None, 20)                220

#  dense_1 (Dense)             (None, 15)                315

#  dense_2 (Dense)             (None, 10)                160

#  dense_3 (Dense)             (None, 1)                 11

# =================================================================
# Total params: 1,096
# Trainable params: 1,096
# Non-trainable params: 0
# _________________________________________________________________

