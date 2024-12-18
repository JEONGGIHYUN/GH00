import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([[1,2,3,4,5],
               [6,7,8,9,10]])
# x = np.array([[1,6],[2,7],[3,8],[4,9],[5,10]])
y = np.array([1,2,3,4,5])

# 열과 행을 바꿀때에는 x = x.T 또는 x = np.transpose(x) 를 사용하면 된다.
x = x.T



print(x.shape) # (5, 2)
print(y.shape) # (5,)


#2. 모델구성
model = Sequential()
model.add(Dense(1000, input_dim=2))
model.add(Dense(250))
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=100)

#4. 평가 예측
loss = model.evaluate(x,y)
results = model.predict([[6,11]])
print('로스 :', loss)
print('[6, 11]의 예측값 :',  results)
