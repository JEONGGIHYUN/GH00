# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Reshape, Conv1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


path = 'C:/ai5/_data/kaggle/santander-customer-transaction-prediction/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)

test_csv = pd.read_csv(path + 'test.csv', index_col=0)

submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

x = train_csv.drop(['target'], axis=1)
# print(x.shape) # (200000, 200)

y = train_csv['target']
# print(y.shape) # (200000,)``

# x.boxplot() # hour_bef_visibility
# x.plot.box()
# plt.show()

x = np.log1p(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=3, stratify=y) #2845

############################### y 로그 변환 ########################
# y_train = np.log1p(y_train)
# y_test = np.log1p(y_test)

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