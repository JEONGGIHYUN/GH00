import numpy as np
from sklearn.datasets import load_breast_cancer
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR) 
print(datasets.feature_names)
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (569, 30) (569, )
print(type(x)) # <class 'numpy.ndarray'>

# 0과 1의 갯수가 몇개인지 찾기
print(np.unique(y, return_counts=True)) # 2진 분류의 갯수를 확인하는 이유 : 데이터의 불균형을 확인해야 한다. 
# (array([0, 1]), array([212, 357], dtype=int64))
# print(y.value_counts()) # 에러 코드
print(pd.DataFrame(y).value_counts()) # 올바른 코드
# 1    357
# 0    212
# print(pd.Series(y).value_counts()) # 올바른 코드와 같다
# print(pd.value_counts(y)) # 올바른 코드와 같다 

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=4521, shuffle=True)

print(x_train.shape, y_train.shape) # (512, 30) (512,)
print(x_test.shape, y_test.shape) # (57, 30) (57,)

################################################
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
scaler = RobustScaler()

scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)
################################################
print(x_train)
print(np.min(x_train), np.max(x_train)) #  0.0 4254.0
print(np.min(x_test), np.max(x_test)) # 0.0 3432.0

#2. 모델 구성

#3. 컴파일 훈련

#4. 평가 예측

print('================== 1. save.model 출력====================')
model = load_model('C:/TDS/ai5/study/_save/keras30_mcp/k30_06_0726_1944_0029-0.0547.hdf5')

loss = model.evaluate(x_test, y_test)


y_pred = model.predict(x_test)
print(y_pred)
r2 = r2_score(y_test, y_pred)
y_pred = np.round(y_pred)

print(y_pred)

accuracy_score = accuracy_score(y_test, y_pred)


print('로스 :', loss)
print('acc_score :', accuracy_score)
print('r2 스코어 :', r2)

# 로스 : [0.06152462959289551, 0.9122806787490845]
# acc_score : 0.9122807017543859
# 걸린 시간 : 2.15 초
# r2 스코어 : 0.7418137435127825

# 로스 : [0.05259425565600395, 0.9356725215911865]
# acc_score : 0.935672514619883
# 걸린 시간 : 2.06 초
# r2 스코어 : 0.779289816

# 로스 : [0.05480422079563141, 0.9181286692619324]
# acc_score : 0.9181286549707602
# 걸린 시간 : 4.28 초
# r2 스코어 : 0.770015758141759

# MaxAbsScaler
# 로스 : [0.060993362218141556, 0.9122806787490845]
# acc_score : 0.9122807017543859
# 걸린 시간 : 2.34 초
# r2 스코어 : 0.7440432083627748

# 로스 : [0.06187659874558449, 0.9181286692619324]
# acc_score : 0.9181286549707602
# 걸린 시간 : 2.49 초
# r2 스코어 : 0.7403367392995412

# RobustScaler
# 로스 : [0.04882717505097389, 0.9356725215911865]
# acc_score : 0.935672514619883
# 걸린 시간 : 2.22 초
# r2 스코어 : 0.7950982319050518

# RobustScaler
# 로스 : [0.07215014845132828, 0.8947368264198303]
# acc_score : 0.8947368421052632
# 걸린 시간 : 1.08 초
# r2 스코어 : 0.6972241082467483

# 로스 : [0.058172158896923065, 0.9239766001701355]
# acc_score : 0.9239766081871345
# 걸린 시간 : 3.35 초
# r2 스코어 : 0.7558822890168698