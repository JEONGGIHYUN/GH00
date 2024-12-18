import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

tf.random.set_seed(333)
np.random.seed(333)

#1. 데이터

# x = np.array([1,2,3,4,5])
# y = np.array([1,2,3,4,5])

x = np.array([1])
y = np.array([1])

#2. 모델
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

###########################################################
# model.trainable = False         # 동결 ★★★★★
model.trainable = True         # 안동결 ★★★★★ 디폴트
###########################################################
print('##########################################')
print(model.weights)
print('##########################################')
###########################################################
### [실습] 위에 weights를 손계산해서 1을 만들기 ###

#3. 컴파일 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x,y, batch_size=1,epochs=2850, verbose=0)

#4. 평가, 예측
y_pred = model.predict(x)
print(y_pred)

print(model.weights)
