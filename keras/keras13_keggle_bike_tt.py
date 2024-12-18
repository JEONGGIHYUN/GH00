import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


#1. 데이터
path = 'C:/TDS/ai5/_data/bike-sharing-demand/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
sampleSubmission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)

x = train_csv.drop(['casual','registered','count'], axis=1)

y = train_csv[['casual','registered']]
print(y.shape)

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
model.add(Dense(2, activation='linear'))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=1000)

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)

y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape) #(6493, 2) 넘파이 데이터로 제공 된다.

# print('test_csv타입 :',type(test_csv))
# print('y_submit타입 :', type(y_submit))

test2_csv = test_csv 
print(test2_csv.shape) #(6493, 8)

test2_csv[['casual','registered']] = y_submit
print(test2_csv)  #  [6493 x 10]

test2_csv.to_csv(path + 'test2.csv')









