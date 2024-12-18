# https://dacon.io/competitions/official/236068/overview/description

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import time


#1. 데이터
path = './_data/dacon/diabetes/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
# print(train_csv) # [652 rows x 9 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=0)
# print(test_csv) # [116 rows x 8 columns]

submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)
# print(submission_csv)

# print(train_csv.shape) # (652, 9)
# print(test_csv.shape) # (116, 8)
# print(submission_csv.shape) # (116, 1)

# print(train_csv.columns)
# Index(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
    #    'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'],
    #   dtype='object')

x = train_csv.drop(['Outcome'], axis=1)

y = train_csv['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=3421)

# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)

################################################
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
scaler = RobustScaler()

scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)
################################################
# # print(x_train)
# print(np.min(x_train), np.max(x_train)) # 
# print(np.min(x_test), np.max(x_test)) #

#2. 모델 구성

#3. 컴파일 훈련

#4. 평가 예측

print('================== 1. save.model 출력====================')
model = load_model('C:/TDS/ai5/study/_save/keras30_mcp/k30_07_0726_1945_0020-0.1889.hdf5')

loss = model.evaluate(x_test, y_test)
y_pred = np.round(model.predict(x_test))
y_submit = model.predict(test_csv)
print(y_pred)
acc = accuracy_score(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# submission_csv['Outcome'] = np.round(y_submit)
print('로스 :', loss)

# submission_csv.to_csv(path + 'submission_0722_1635.csv')

# 로스 : [0.2036416232585907, 0.6989796161651611]

# 로스 : [0.20923519134521484, 0.6887755393981934]

# 로스 : [0.21842415630817413, 0.6938775777816772]

# MaxAbsScaler
# 로스 : [0.30629029870033264, 0.6224489808082581]

# 로스 : [0.2141127735376358, 0.6836734414100647]

# RobustScaler
# 로스 : [0.3571428656578064, 0.6428571343421936]

# 로스 : [0.21126089990139008, 0.6938775777816772]