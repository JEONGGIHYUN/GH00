from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,4,5,6])

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=10000)

#4. 평가, 예측
loss = model.evaluate(x,y) 
print('로스 :', loss) # 오차 범위가 몇 인지 확인
result = model.predict(np.array([1,2,3,4,5,6,7])) #1234567의 예측값을 확인
print('6의 예측값 :', result) #1234567의 예측값을 도출
