import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y = np.array([1,2,3,4,5,6,7,8,9,10])

x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])

x_test = np.array([8,9,10])
y_test = np.array([8,9,10])


#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=1,
          verbose=4) # verbose의 디폴트 값은 1 이다.
# verbose0 : 침묵
# verbose1 : 디폴트
# verbose2 : 프로그래스바 삭제
# verbose4 : 에포만 나온다

#4. 평가 예측
print('==========================================')
loss = model.evaluate(x_test, y_test)
results = model.predict([11])
print('로스 :', loss)
print('11의 결과값 :', results)

# 7/7 [==============================] - 0s 0s/step - loss: 8.1262e-07
# ==========================================
# 1/1 [==============================] - 0s 34ms/step - loss: 2.9720e-06
# 로스 : 2.9719840313191526e-06
# 11의 결과값 : [[11.002506]]

#7/7 [==============================] - 0s 0s/step - loss: 1.7621e-12
#==========================================
#1/1 [==============================] - 0s 46ms/step - loss: 9.4739e-12
#로스 : 9.473903432588582e-12
#11의 결과값 : [[10.999995]]

#7/7 [==============================] - 0s 332us/step - loss: 4.9766e-11
#==========================================
#1/1 [==============================] - 0s 32ms/step - loss: 1.9433e-10
#로스 : 1.9432870590474494e-10
#11의 결과값 : [[11.00002]]



