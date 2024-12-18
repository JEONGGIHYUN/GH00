# https://www.kaggle.com/competitions/bike-sharing-demand/data

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time


#1. 데이터
# path = './_data/bike_sharing_demand/' # 상대 경로 방식

# * 3개 다 같은 방식이다 *
# path = 'C:\\TDS\\ai5\\_data\\bike-sharing-demand' # 절대 경로 방식
# path = 'C://TDS//ai5//_data//bike-sharing-demand//' 
path = 'C:/TDS/ai5/_data/bike-sharing-demand/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)

test_csv = pd.read_csv(path + 'test.csv', index_col=0)

sampleSubmission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)

###### x와 y를 분리 ########
x = train_csv.drop(['casual','registered','count'], axis=1)
print(x.shape) # (10886, 8)
# print(x)

y = train_csv['count']
print(y.shape)

# print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.87, random_state=57543)

#2. 모델 구성
model = Sequential()
model.add(Dense(16, activation='relu', input_dim=8))
model.add(Dense(256, activation='relu'))
model.add(Dense(512))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))


#3. 컴파일 훈련

start_time = time.time()

model.compile(loss='mse', optimizer='adam')

hist = model.fit(x_train, y_train, epochs=30, batch_size=1000,
          verbose=1,
          validation_split=0.3)

end_time = time.time()

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor= 'val_loss',
    mode = 'min',
    patience=10,
    restore_best_weights=True
)

#4. 평가 예측
loss = model.evaluate(x_test, y_test)

results = model.predict(x_test)

# print(results)


# y_submit = model.predict(test_csv)
# # print(y_submit)
# # print(y_submit.shape)

# sampleSubmission['count'] = y_submit
# print(sampleSubmission)
# print(sampleSubmission.shape)
r2 = r2_score(y_test, results)

print('로스 :', loss)
print('r2 :', r2)
# sampleSubmission.to_csv(path + 'submission_0717_1748.csv')

print('소요시간 :', round(end_time - start_time),'초')
print('==========hist=========')
print(hist)
print('==========hist.history=========')
print(hist.history)
print('==========lose=========')
print(hist.history['loss'])
print('==========val_loss=========')
print(hist.history['val_loss'])
print('===================================')

import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', label='loss', marker='.')
plt.plot(hist.history['val_loss'], c='green', label='val_loss', marker='.')
plt.legend(loc='upper right')
plt.title('케글 바이크')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.show()




