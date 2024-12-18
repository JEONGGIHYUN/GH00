import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array(range(10)) # range()를 사용할때는 마지막 숫자는 없다고 생각하는게 좋다 range의 사용방법은 range(1, 10)또는 range(10, )가 있다

print(x) # [0,1,2,3,4,5,6,7,8,9]
print(x.shape) # (10, )

x = np.array(range(1, 11))
print(x)
print(x.shape)

x = np.array([range(10), range(21, 31), range(201,211)])
x = x.T
print(x)
print(x.shape) #(10, 3)
y = np.array([1,2,3,4,5,6,7,8,9,10])

#2. 모델 구성
model = Sequential()
model.add(Dense(1000, input_dim=3))
model.add(Dense(500))
model.add(Dense(250))
model.add(Dense(25))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=1000)

#4. 평가 예측
loss = model.evaluate(x,y)
result = model.predict([[10,31,211]])
print('로스 :', loss)
print('11의 예측값은 :', result)




#[실습] 
