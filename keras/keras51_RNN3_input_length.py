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
# model.add(SimpleRNN(units=21, input_shape=(3,1), activation='relu')) # timesteps, features
# model.add(SimpleRNN(units=21, input_length=3, input_dim=1, activation='relu'))
model.add(SimpleRNN(units=21, input_dim=1, input_length=3, activation='relu')) # 가독성이 덜어져 timesteps(length)와 features(dim)를 바꿔서 입력하지 않는다.
model.add(Dense(20)) # RNN은 Dense와 바로 연결이 가능하다.
model.add(Dense(15, activation='relu'))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    verbose=1,
    restore_best_weights=True
)

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100,
          callbacks=[es],
          validation_split=0.3
          )

#4. 평가, 예측
results = model.evaluate(x, y)
print('loss :', results)

x_pred = np.array([8,9,10]).reshape(1, 3, 1) # [[8],[9],[10]]
y_pred = model.predict(x_pred)
y_pred = np.round(y_pred,1)

print('[8,9,10]의 결과 :', y_pred)


