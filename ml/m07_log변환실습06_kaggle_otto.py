# https://www.kaggle.com/c/otto-group-product-classification-challenge/leaderboard?tab=public
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten,Conv1D, Reshape
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

path = 'C:/ai5/_data/kaggle/otto-group-product-classification-challenge/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)

test_csv = pd.read_csv(path + 'test.csv', index_col=0)

submission_csv = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)

# train_csv.info()
# test_csv.info()
# print(train_csv['target'].value_counts())
# train_csv['target'] = train_csv['target'].replace({'Class_1' : 1, 'Class_1' : 1, 'Class_2' : 2, 'Class_3' : 3, 'Class_4' : 4, 'Class_5' : 5, 'Class_6' : 6, 'Class_7' : 7, 'Class_8' : 8, 'Class_9' : 9, })


############################################
x = train_csv.drop(['target'], axis=1)
# print(x.shape) # (61878, 93)

y = train_csv['target']
# print(y.shape) # (61878,)
###################################
y = pd.get_dummies(y)
# print(y)
# print(y.shape)

# x.boxplot() # hour_bef_visibility
# x.plot.box()
# plt.show()


x = np.log1p(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=3, shuffle=True, stratify=y)

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
# score : 0.2682922659034375
# r2 score : 0.2682922659034375

# 로그 변환 x만 한 후
# score : 0.26831773207583426
# r2 score : 0.26831773207583426

# 로그 변환  x,y둘 다
# score : 0.2683566626065261
# r2 score : 0.2683566626065261