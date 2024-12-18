import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([range(10)])
y = np.array([[1,2,3,4,5,6,7,8,9,10]
              ,[10,9,8,7,6,5,4,3,2,1]
              ,[9,8,7,6,5,4,3,2,1,0]]) 

print(x.shape) #(1, 10)
print(y.shape) #(3, 10)

x = x.T
y = np.transpose(y)

print(x.shape) #(10, 3)
print(y.shape) #(10, 2)

#2. 모델
# [실습] 만들기
# x_predict = [10, 31, 211]
model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(500))
model.add(Dense(250))
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(5))
model.add(Dense(3))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=1000)

#4. 평가 예측
loss = model.evaluate(x,y)  # 훈련을 할때는 evaluate를 x,y값을 사용하면 정확할 수 있으나 x,y값 이외의 것을 사용하면 결과값이 많이 떨어진다.
result = model.predict([[11]])
print('로스 :', loss)
print('11과 0과 -1의 예측값 :', result)

#  로스 : 6.136614653928785e-13
#  11과 0과 -1의 예측값 : [[11.999999  -1.0000001 -1.9999989]]

