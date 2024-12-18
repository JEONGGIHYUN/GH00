# https://www.kaggle.com/competitions/playground-series-s4e1/overview


import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder

#1. 데이터
path = "C:\\TDS\\ai5\\_data\\kaggle\\playground-series-s4e1\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
mission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

print(train_csv.shape)      # (165034, 13)
print(test_csv.shape)       # (110023, 12)
print(mission_csv.shape)    # (110023, 1)

print(train_csv.columns)
# Index(['CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender', 'Age',
#        'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
#        'EstimatedSalary', 'Exited'],

print(train_csv.isnull().sum())   # 결측치가 없다
print(test_csv.isnull().sum())

encoder = LabelEncoder()
train_csv['Geography'] = encoder.fit_transform(train_csv['Geography'])
test_csv['Geography'] = encoder.fit_transform(test_csv['Geography'])
train_csv['Gender'] = encoder.fit_transform(train_csv['Gender'])
test_csv['Gender'] = encoder.fit_transform(test_csv['Gender'])

###############################################
train_csv = train_csv.drop(['CustomerId', 'Surname'], axis=1)

test_csv = test_csv.drop(['CustomerId', 'Surname'], axis=1)

from sklearn.preprocessing import MinMaxScaler

train_scaler = MinMaxScaler()

train_csv_copy = train_csv.copy()

train_csv_copy = train_csv_copy.drop(['Exited'], axis = 1)

train_scaler.fit(train_csv_copy)

train_csv_scaled = train_scaler.transform(train_csv_copy)

train_csv = pd.concat([pd.DataFrame(data = train_csv_scaled), train_csv['Exited']], axis = 1)

test_scaler = MinMaxScaler()

test_csv_copy = test_csv.copy()

test_scaler.fit(test_csv_copy)

test_csv_scaled = test_scaler.transform(test_csv_copy)

test_csv = pd.DataFrame(data = test_csv_scaled)
###############################################

x = train_csv.drop(['Exited'], axis=1)
print(x)                            # [165034 rows x 10 columns]
y = train_csv['Exited']
print(y.shape)                      # (165034,)

print(np.unique(y, return_counts=True))     
# (array([0, 1], dtype=int64), array([130113,  34921], dtype=int64))
print(pd.DataFrame(y).value_counts())
# 0         130113
# 1          34921

from sklearn.preprocessing import MinMaxScaler
scalar=MinMaxScaler()
x[:] = scalar.fit_transform(x[:])

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=7575)

#2. 모델구성
model = Sequential()
model.add(Dense(32, input_dim=10))
model.add(Dense(64, activation='relu'))
model.add(Dense(148, activation='relu'))
model.add(Dense(168, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
start_time = time.time()
es = EarlyStopping(monitor='val_loss',   # arlyStopping 정의
                   mode = 'min',               # 모르면 auto
                   patience=10,
                   restore_best_weights=True,)

hist = model.fit(x_train, y_train, epochs=4680, batch_size=4220, validation_split=0.2, 
                 callbacks=[es])
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("로스 : ", loss[0])
print("ACC : ", round(loss[1], 3))

y_predict = model.predict(x_test)
print(y_predict[:20])       # y' 결과
y_predict = np.round(y_predict)
print(y_predict[:20])       # y' 반올림 결과

accuracy_score = accuracy_score(y_test, y_predict)
print("acc score : ", accuracy_score)
print("걸린 시간 : ", round(end_time - start_time, 2), "초")

y_submit = model.predict(test_csv)
print(y_submit.shape)       # (110023, 1)

y_submit = np.round(y_submit)
mission_csv['Exited'] = y_submit
mission_csv.to_csv(path + "sample_submission_0723_1205.csv")

'''
32 16 16 16 16 1
train_size=0.8, random_state=3434 / epochs=100, batch_size=16, validation_split=0.2
acc score :  0.7709923664122137

++++++++++++++++++++++++++++++
random_state=6666
random_state=1866
random_state=1186
'''

# random_state=7777
# 로스 :  0.3238998055458069

# random_state=7575
# model.add(Dense(32, input_dim=10))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# 로스 :  0.32291173934936523

# model.add(Dense(32, input_dim=10))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(168, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# epochs=1110
# 로스 :  0.3226291239261627

# epochs=1300

# random_state=7575
# model.add(Dense(32, input_dim=10))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(148, activation='relu'))
# model.add(Dense(168, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# epochs=2680, batch_size=4220
# 로스 :  0.32392868399620056

# epochs=4680
# 로스 :  0.3231939375400543