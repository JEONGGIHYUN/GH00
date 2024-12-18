import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
# *
import time 
# *
#1. 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.65, random_state=133, shuffle=True)

#2. 모델 구성
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(51))
model.add(Dense(25))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))


#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
# *
start_time = time.time()
# *
model.fit(x_train,y_train, epochs=100, batch_size=2,
          verbose=1,
          validation_split=0.3) # train 데이터의 0.3 을 분리하여 사용 하겠다.
# *
end_time = time.time()
# *

#4. 평가 예측
loss = model.evaluate(x_test, y_test, verbose=0)
results = model.predict([18])
print('로스 :', loss)
print('18의 예측값 :', results)
# *
print('걸린시간 :',round(end_time - start_time, 2), '초') 
# *
