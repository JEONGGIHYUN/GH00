from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston

#1. 데이터
dataset = load_boston()

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=1,shuffle=True)



#2. 모델 구성
model = Sequential()
model.add(Dense(150, input_dim=13))
model.add(Dense(30))
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(7))
model.add(Dense(1))

#3. 컴파일 훈련

start_time = time.time()

model.compile(loss='mse', optimizer='adam')

hist = model.fit(x_train, y_train, epochs=50,  batch_size=32,
                 verbose=2,
                 validation_split=0.3)

end_time = time.time()

#4. 평가 예측

loss = model.evaluate(x_test, y_test)

results = model.predict(x_test)

print('로스 :', loss)

print('보스턴 집값 :', results)

r2 = r2_score(y_test, results)

print('r2스코어 :', r2)

print('소요시간 :', round(end_time - start_time), '초')

# # *
# print('================ hist ================')
# print(hist)
# print('================ hist.history ================')
# print(hist.history)
# print('================ loss ================')
# print(hist.history['loss'])
# print('================ val_loss ================')
# print(hist.history['val_loss'])
# print('==============================================')
# # *

# # *
# import matplotlib.pyplot as plt
# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], c='red', label='loss', marker='.')
# plt.plot(hist.history['val_loss'], c='green', label='val_loss', marker='.')
# plt.legend(loc='upper right')
# plt.title('보스턴 loss')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.grid()
# plt.show()
# # *