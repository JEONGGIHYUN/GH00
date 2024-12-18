# https://www.kaggle.com/c/playground-series-s4e1/overview

import numpy as np
from sklearn.datasets import load_breast_cancer
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
from pandas import DataFrame

#1. 데이터
path = './_data/kaggle/playground-series-s4e1/'

train_csv = pd.read_csv(path + 'train.csv', index_col=[0,1,2])
# print(train_csv) # [165034 rows x 13 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=[0,1,2]) 
# print(test_csv) #  [110023 rows x 12 columns]

submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0) 
# print(submission_csv) # [110023 rows x 1 columns]

# print(train_csv.shape) # (165034, 13)
# print(test_csv.shape) # (110023, 12)
# print(submission_csv.shape) # (110023, 1)

# print(train_csv.columns)
# Index(['CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender', 'Age',
#        'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
#        'EstimatedSalary', 'Exited'],
#       dtype='object')
'''
df = pd.DataFrame(train_csv)

# train_csv = train_csv['Geography'].str.replace('France', '1')
df = df.replace({'Geography':'France'}, '0')
df = df.replace({'Geography':'Germany'}, '1')
df = df.replace({'Geography':'Spain'}, '2')

df = df.replace({'derGen':'Male'}, '1')
df = df.replace({'derGen':'Female'}, '0')
'''
geography_mapping = {'France': 0, 'Germany': 1, 'Spain': 2}
derGen_mapping = {'Male': 1, 'Female': 0}

train_csv['Geography'] = train_csv['Geography'].map(geography_mapping)
train_csv['Gender'] = train_csv['Gender'].map(derGen_mapping)

test_csv['Geography'] = test_csv['Geography'].map(geography_mapping)
test_csv['Gender'] = test_csv['Gender'].map(derGen_mapping)


x = train_csv.drop(['Exited'], axis=1)
y = train_csv['Exited']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=4524)

# print(x_train.shape, y_train.shape) # (115523, 11) (115523, 2)
# print(y_train.shape, y_test.shape) # (115523, 2) (49511, 2)

# ###############################################
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()

# scaler.fit(x_train)
# scaler.transform(x_train)
# scaler.transform(x_test)
# ################################################
# # print(x_train)
# print(np.min(x_train), np.max(x_train)) # 
# print(np.min(x_test), np.max(x_test)) #



#2. 모델 구성
model = Sequential()
model.add(Dense(500, input_dim=10))
model.add(Dense(250,))
model.add(Dense(125,))
model.add(Dense(75,))
model.add(Dense(25,activation='sigmoid'))
model.add(Dense(1,activation='sigmoid'))

#3. 컴파일 훈련
start = time.time()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience=10,
    restore_best_weights=True
)

hist = model.fit(x_train, y_train, epochs=100,
                 verbose=1,
                 validation_split=0.4,
                 batch_size=5000,
                 callbacks=[es])

end = time.time()

# #4. 평가 예측
loss = model.evaluate(x_test, y_test)
y_pred = np.round(model.predict(x_test))
y_submit = model.predict(test_csv)

acc = accuracy_score(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

submission_csv['Exited'] = np.round(y_submit)
print('로스 :', loss)
print('acc_score :', accuracy_score)
print('걸린 시간 :', round(end - start, 2), '초')
print('r2 스코어 :', r2)

# submission_csv.to_csv(path + 'submission_0722_19_22.csv')

# 로스 : [0.5048302412033081, 0.7903092503547668]
# acc_score : <function accuracy_score at 0x000001D52C1F4670>
# 걸린 시간 : 10.57 초
# r2 스코어 : -0.2653275064530145

# 로스 : [0.5135741233825684, 0.7903092503547668]
# acc_score : <function accuracy_score at 0x0000019D1F914670>
# 걸린 시간 : 9.56 초
# r2 스코어 : -0.2653275064530145
