import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([range(10), range(21, 31), range(201, 211)])
y = np.array([[1,2,3,4,5,6,7,8,9,10]
              ,[10,9,8,7,6,5,4,3,2,1]]) 

print(x.shape) #(3, 10)
print(y.shape) #(2, 10)

x = x.T
y = np.transpose(y)

print(x.shape) #(10, 3)
print(y.shape) #(10, 2)

#2. 모델
# [실습] 만들기
# x_predict = [10, 31, 211]
model = Sequential()
model.add(Dense(1000, input_dim=3))
model.add(Dense(500))
model.add(Dense(250))
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(5))
model.add(Dense(2))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=1000)

#4. 평가 예측
loss = model.evaluate(x,y)
result = model.predict([[10,31,211]])
print('로스 :', loss)
print('11과 0의 예측값 :', result)

