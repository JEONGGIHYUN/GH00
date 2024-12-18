from tensorflow.keras.models import Sequential #텐서플로우 케라스에 있는 모델 순차적인
from tensorflow.keras.layers import Dense #텐서플로우 케라스에 있는 레이어 조밀한
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])


#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))

epochs = 2000
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

#epochs : 10000
#로스 : 0.3800000250339508
#6번째 숫자의 예측값 : [[5.7000003]]

#epochs : 100
#로스 : 0.3941521942615509
#6번째 숫자의 예측값 : [[5.709633]]

