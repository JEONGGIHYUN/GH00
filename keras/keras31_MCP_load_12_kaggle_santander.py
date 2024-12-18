# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time



path = 'C:/TDS/ai5/_data/kaggle/santander-customer-transaction-prediction/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)

test_csv = pd.read_csv(path + 'test.csv', index_col=0)

submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

x = train_csv.drop(['target'], axis=1)
# print(x.shape) # (200000, 200)

y = train_csv['target']
# print(y.shape) # (200000,)``

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=3, stratify=y) #2845

##############################################
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
scaler = RobustScaler()

scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)
###############################################
# print(x_train)
# print(np.min(x_train), np.max(x_train)) #
# print(np.min(x_test), np.max(x_test)) # 

#2. 모델 구성

#3. 컴파일 훈련

#4. 예측 평가

print('================== 1. save.model 출력====================')
model = load_model('C:/TDS/ai5/study/_save/keras30_mcp/k30_12_0726_1950_0013-0.2539.hdf5')

loss = model.evaluate(x_test, y_test)
y_predict = np.around(model.predict(x_test))
y_submit = model.predict(test_csv)
accuracy_score = accuracy_score(y_test, y_predict)

submission_csv['target'] =(y_submit)

print('loss :', loss)
print('acc score :', accuracy_score)

# submission_csv.to_csv(path + 'submission_0725_16_51.csv')
'''
loss : [1.3962657451629639, 0.888949990272522]
acc score : 0.88895
time : 138.13 초

loss : [1.2887171506881714, 0.8784999847412109]
acc score : 0.8785
time : 138.51 초

loss : [1.2086162567138672, 0.8866333365440369]
acc score : 0.8866333333333334
time : 136.52 초

loss : [1.214404821395874, 0.8827499747276306]
acc score : 0.88275
time : 140.57 초

MaxAbsScaler
loss : [1.0676288604736328, 0.885016679763794]
acc score : 0.8850166666666667
time : 135.4 초

loss : [1.1891380548477173, 0.8880166411399841]
acc score : 0.8880166666666667
time : 100.43 초

RobustScaler
loss : [1.0957788228988647, 0.8907666802406311]
acc score : 0.8907666666666667
time : 67.54 초

loss : [1.0016027688980103, 0.8838333487510681]
acc score : 0.8838333333333334
time : 68.88 초

'''