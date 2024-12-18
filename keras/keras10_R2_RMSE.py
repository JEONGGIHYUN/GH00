# 09_2 카피

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.6, random_state=112)

#2. 모델
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=5)

#4. 평가 예측
loss = model.evaluate(x_test, y_test)

print('로스 :', loss)



y_predict = model.predict(x_test)
#검색 R2
r2 = r2_score(y_test, y_predict)
print('=====================================================================')
print('r2스코어 :', r2)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print('RMSE :', rmse)