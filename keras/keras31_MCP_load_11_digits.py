from sklearn.datasets import load_digits
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

x, y = load_digits(return_X_y=True) # 사이킷런에서 사용 가능한 방식이다.

# print(x)
# print(y)
# print(x.shape, y.shape) #(1797, 64) (1797,)

print(pd.value_counts(y, sort=False))
# 0    178
# 1    182
# 2    177
# 3    183
# 4    181
# 5    182
# 6    181
# 7    179
# 8    174
# 9    180
# dtype: int64

y = pd.get_dummies(y)
print(y)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1346, train_size=0.75, shuffle=True, stratify=y)

###############################################
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
scaler = RobustScaler()

scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)
################################################
# print(x_train)
print(np.min(x_train), np.max(x_train)) # 0.0 16.0
print(np.min(x_test), np.max(x_test)) # 0.0 16.0


#2. 모델 구성

#3. 컴파일 훈련

#4. 예측 평가

print('================== 1. save.model 출력====================')
model = load_model('C:/TDS/ai5/study/_save/keras30_mcp/k30_11_0726_1949_0153-0.0824.hdf5')

loss = model.evaluate(x_test, y_test)
y_predict = np.around(model.predict(x_test))
accuracy_score = accuracy_score(y_test, y_predict)

print('loss :', loss)
print('acc score :', accuracy_score)


'''
loss : [0.16231749951839447, 0.9733333587646484]
acc score : 0.9733333333333334
time : 49.37 초

loss : [0.07097781449556351, 0.9777777791023254]
acc score : 0.9777777777777777
time : 49.53 초

loss : [0.1270405799150467, 0.9755555391311646]
acc score : 0.9755555555555555
time : 49.08 초

MaxAbsScaler
loss : [0.13909640908241272, 0.9733333587646484]
acc score : 0.9733333333333334
time : 49.13 초

loss : [0.24305672943592072, 0.9688888788223267]
acc score : 0.9688888888888889
time : 54.53 초

RobustScaler
loss : [0.21911989152431488, 0.9644444584846497]
acc score : 0.9644444444444444
time : 49.56 초

loss : [0.2355182021856308, 0.9733333587646484]
acc score : 0.9711111111111111
time : 49.4 초

'''