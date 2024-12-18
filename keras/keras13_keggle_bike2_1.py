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
# sampleSubmission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)

# x와 y를 분리
x = train_csv.drop(['casual','registered','count'], axis=1)

y = train_csv[['casual','registered']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.87, random_state=1354)



#2. 모델 구성
model = Sequential()
model.add(Dense(15, activation='relu', input_dim=8))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='linear'))


#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=5, batch_size=500)

#4. 데이터 수정
y_submit = model.predict(test_csv)

casual_predict = y_submit[:,0]
registered_predict = y_submit[:,1]



test_csv = test_csv.assign(casual=y_submit[:,0], registered = y_submit[:,1])

test_csv.to_csv(path + 'test_01.csv')



#5. x와 y를 분리
#6. 모델 구성
#7. 컴파일 훈련
#4. 평가 예측