from tensorflow.keras.models import Sequential #텐서플로우 케라스에 있는 모델 순차적인
from tensorflow.keras.layers import Dense #텐서플로우 케라스에 있는 레이어 조밀한
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6]) # 데이터 ([]) 안에 들어있는 데이터는 한 개의 데이터로 취급한다 ([][]) 이렇게 되어 있는 데이터는 두 개의 데이터로 취급한다.
y = np.array([1,2,3,5,4,6]) 

#[실습] 레이어의 깊이와 노드의 갯수를 이용해서 최소의 loss를 만들어
# 에포는 100으로 고정, 건들지말것
# 로스 기준 0.33 이하


#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=1)) #inptu_dim=1 : 인풋에 해당하는 레이어 (?, inpit_dim=1) : ? 는 인풋에 해당하는 레이어의 아웃풋
model.add(Dense(50)) #위에 input_dim을 사용하면 밑의 아웃풋은 히든 레이어가 됨 으로 input_dim을 묵음처리하고 사용해도 된다.
model.add(Dense(25))  
model.add(Dense(5))  
model.add(Dense(3))    
model.add(Dense(2))  
model.add(Dense(1))  
  


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
