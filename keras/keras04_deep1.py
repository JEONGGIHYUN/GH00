from tensorflow.keras.models import Sequential #텐서플로우 케라스에 있는 모델 순차적인
from tensorflow.keras.layers import Dense #텐서플로우 케라스에 있는 레이어 조밀한
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

#[실습] 레이어의 깊이와 노드의 갯수를 이용해서 [6]을 만들어
# 에포는 100으로 고정, 건들지말것
# 소수 네째자리까지 맞추면 합격. 예: 6.0000 또는 5.9999


#2. 모델구성
model = Sequential()
model.add(Dense(1000, input_dim=1)) #inptu_dim=1 : 인풋에 해당하는 레이어 (?, inpit_dim=1) : ? 는 인풋에 해당하는 레이어의 아웃풋
model.add(Dense(250, input_dim=1000))
model.add(Dense(50, input_dim=250))  
model.add(Dense(25, input_dim=50))  
model.add(Dense(5, input_dim=25))    
model.add(Dense(3, input_dim=5))  
model.add(Dense(1, input_dim=3))  
  


epochs = 100
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=epochs)

#4. 평가, 예측
loss = model.evaluate(x,y)
print('==========================')
print('epochs :', epochs)
print('로스 :', loss)
result = model.predict(np.array([6]))
print('6번째 숫자의 예측값 :', result)
