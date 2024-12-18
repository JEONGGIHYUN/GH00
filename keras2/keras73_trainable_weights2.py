import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

tf.random.set_seed(333)
np.random.seed(333)

#1. 데이터

x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

x = np.array([1])
y = np.array([1])

#2. 모델
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

###########################################################
model.trainable = False         # 동결 ★★★★★
# model.trainable = True         # 안동결 ★★★★★ 디폴트
###########################################################
print('##########################################')
print(model.weights)
print('##########################################')

# [<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[ 0.13603318, -0.03480017,  0.7743634 ]], dtype=float32)>, <tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, <tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy=
# array([[-0.92561173,  0.8256177 ],
    #    [ 0.6200088 ,  1.0182774 ],
    #    [-0.5191052 , -0.6304303 ]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>, <tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=
# array([[-0.02628279],
    #    [-1.074922  ]], dtype=float32)>, <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]
##########################################


#3. 컴파일 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x,y, batch_size=2,epochs=2870, verbose=0)

#4. 평가, 예측
y_pred = model.predict(x)
print(y_pred)

