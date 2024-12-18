# https://dacon.io/competitions/official/235576/overview/description 대회 주소

import numpy as np
import pandas as pd # 인덱스와 칼럼을 분리하는데 사용하는 함수
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
import matplotlib.pyplot as plt

#1. 데이터
path = './_data/따릉이/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)


test_csv = pd.read_csv(path + 'test.csv', index_col=0) 


submission_csv = pd.read_csv(path + 'submission.csv', index_col=0) # [715 rows x 1 columns]

print(train_csv.info())

############ 결측치 처리 1. 삭제 ##############
# train_csv.isnull().sum()
print(train_csv.isna().sum())

train_csv = train_csv.dropna() 
print(train_csv.isna().sum())
print(train_csv) # [1328 rows x 10 columns]

print(test_csv.info())

test_csv = test_csv.fillna(test_csv.mean()) # 결측치 채우기 
print(test_csv.info())

x = train_csv.drop(['count'], axis=1)
print(x) # [1328 rows x 9 columns]

y = train_csv['count']
print(y.shape) # (1328, )

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=4343)

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience=10,
    restore_best_weights=True
)

#2. 모델 구성
model = Sequential()
model.add(Dense(500, input_dim=9))
model.add(Dense(250))
model.add(Dense(125))
model.add(Dense(72.5))
model.add(Dense(50))
model.add(Dense(1))



#3. 컴파일 훈련
start_time = time.time()

model.compile(loss='mse', optimizer='adam')

hist = model.fit(x_train, y_train, epochs=1000,
          verbose=0,
          validation_split=0.5)

end_time = time.time()

#4. 평가 예측
loss = model.evaluate(x_test, y_test)

results = model.predict(x_test)
# r2 = r2_score(y_test,  results)
# print('로스 :', loss)

y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape) # (715, 1)


######### submission.csv 만들기 // count 클럼에 값만 넣어 주면 된다. ####
submission_csv['count'] = y_submit
print(submission_csv)
print(submission_csv.shape)

r2 = r2_score(y_test,  results)
print('로스 :', loss)

# *
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False
# *

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', label='loss', marker='.')
plt.plot(hist.history['val_loss'], c='green', label='val_loss', marker='.')
plt.legend(loc='upper right')
plt.title('따릉이')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.show()

#plt를 시각화 하는 방법을 의미한다.