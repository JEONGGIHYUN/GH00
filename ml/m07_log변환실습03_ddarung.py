# https://dacon.io/competitions/official/235576/overview/description 대회 주소

import numpy as np
import pandas as pd # 인덱스와 칼럼을 분리하는데 사용하는 함수
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Reshape, Conv1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


#1. 데이터
path = 'C:/ai5/_data/dacon/따릉이/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
print(train_csv) # [1459 rows x 10 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=0) 
print(test_csv) # [715 rows x 9 columns]

submission_csv = pd.read_csv(path + 'submission.csv', index_col=0) # [715 rows x 1 columns]

############ 결측치 처리 1. 삭제 ##############
train_csv.isnull().sum()

train_csv = train_csv.dropna() 

test_csv = test_csv.fillna(test_csv.mean()) # 결측치 채우기 



x = train_csv.drop(['count'], axis=1)

y = train_csv['count']

# x.boxplot() # hour_bef_visibility
# x.plot.box()
# plt.show()

x['hour_bef_visibility'] = np.log1p(x['hour_bef_visibility']) # 지수변환 np.exp1m

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=4343)

############################### y 로그 변환 ########################
y_train = np.log1p(y_train)
y_test = np.log1p(y_test)

# 2 모델
model = RandomForestRegressor(random_state=1234,
                              max_depth=5,
                              min_samples_split=3,)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가 예측
score = model.score(x_test, y_test)
print('score :', score)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2 score :', r2)

# 로그 변환전
# score : 0.7371601135411457
# r2 score : 0.7371601135411457

# 로그 변환 x만 한 후
# score : 0.7369054013663853
# r2 score : 0.7369054013663853

# 로그 변환  x,y둘 다
# score : 0.7024073031137241
# r2 score : 0.7024073031137241
