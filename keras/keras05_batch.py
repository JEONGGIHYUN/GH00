from tensorflow.keras.models import Sequential #텐서플로우 케라스에 있는 모델 순차적인
from tensorflow.keras.layers import Dense #텐서플로우 케라스에 있는 레이어 조밀한
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6]) # 데이터 ([]) 안에 들어있는 데이터는 한 개의 데이터로 취급한다
y = np.array([1,2,3,4,5,6]) 

#[실습] keras_04 의 가장 좋은 레이어와 노드를 이용하여,
# 최소의 loss를 만들어
# batch_size epochs를 조절하여 
# 로스 기준 0.32 미만


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
model.fit(x,y, epochs=epochs, batch_size=6) # ()안에서 ,를 마지막에 두고 아무것도 쓰지 않아도 실행은 된다. batch_size 를 사용하면 스칼라를 나누어 훈련 할 수 있다.

#4. 평가, 예측
loss = model.evaluate(x,y)
print('==========================')
print('epochs :', epochs)
print('로스 :', loss)
result = model.predict(np.array([7,8,9]))
print('6번째 숫자의 예측값 :', result)

#epochs : 100
#로스 : 0.3238106667995453
#6번째 숫자의 예측값 : [[5.858739]]

#하이퍼 파라미터 튜닝 : 머신러닝 모델의 성능을 향상시키기 위해 모델의 하이퍼 파라미터를 최적화하는 과정