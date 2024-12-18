import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt # matplotlib에서 plot을 plt로 가져오기
#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,7,5,7,8,6,10])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=134, shuffle=True)







#2. 모델
model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(500))
model.add(Dense(250))
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000)

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
results = model.predict(x)
print('로스 :', loss)
print('11의 예측값 :', results)

# matplotlib.pyplot을 plt로 가져오기
import matplotlib.pyplot as plt #데이터의 y=wx + b 의 선을 그어줄때 필요한 데이터 값을 가져오는 함수
plt.scatter(x, y)
plt.plot(x, results, color='red') #데이터가 이어진 선을 그어준다.
plt.show()