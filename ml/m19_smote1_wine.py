import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
# from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
tf.random.set_seed(7777)


#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets['target']

print(x.shape,y.shape)
# (178, 13) (178,)
print(np.unique(y, return_counts=True))
# (array([0, 1, 2]), array([59, 71, 48], dtype=int64))
print(pd.value_counts(y))
# 1    71
# 0    59
# 2    48
# dtype: int64

print(y)
# dtype: int64
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]

x = x[:-40]
y = y[:-40]

print(y)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2]
print(np.unique(y, return_counts=True))
# (array([0, 1, 2]), array([59, 71,  8], dtype=int64))

# y = pd.get_dummies(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=36, shuffle=True, stratify=y)

'''

#2.모델
model = Sequential()
model.add(Dense(10, input_shape=(13,)))
model.add(Dense(3, activation='softmax'))

#3. 컴파일 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
               metrics=['acc'])

model.fit(x_train, y_train, epochs=100, validation_split=0.2)

#4. 평가 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])

# f1_score
y_pred = model.predict(x_test)      # 아그맥으스
y_pred = np.argmax(y_pred, axis=1)
# y_pred = np.argmax(y_pred, axis=1)
# y_test = np.argmax(np.array(y_test), axis=1)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')

print('acc : ', acc)
print('f1 : ', f1)



# loss :  0.39698389172554016
# acc :  0.8571428656578064
# acc :  0.8571428571428571
# f1 :  0.5944444444444444
'''

################# SMOTE 적용 ####################
# pip install imblearn
from imblearn.over_sampling import SMOTE
import sklearn as sk
# print(sk.__version__)
# 통상적으로 train만 증폭을 한다.
# 스모팅 할 데이터가 많을 경우 분할하여 스모팅한다.

print('증폭전 : ', np.unique(y_train, return_counts=True))

smote = SMOTE(random_state=7777)
x_train, y_train = smote.fit_resample(x_train,y_train)
print('증폭후 : ', np.unique(y_train, return_counts=True))
print(pd.value_counts(y_train))

#2.모델
model = Sequential()
model.add(Dense(10, input_shape=(13,)))
model.add(Dense(3, activation='softmax'))

#3. 컴파일 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
               metrics=['acc'])

model.fit(x_train, y_train, epochs=100, validation_split=0.2)

#4. 평가 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])

# f1_score
y_pred = model.predict(x_test)      # 아그맥으스
y_pred = np.argmax(y_pred, axis=1)
# y_pred = np.argmax(y_pred, axis=1)
# y_test = np.argmax(np.array(y_test), axis=1)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')

print('acc : ', acc)
print('f1 : ', f1)

# SMOTE전
# loss :  0.39698389172554016
# acc :  0.8571428656578064
# acc :  0.8571428571428571
# f1 :  0.5944444444444444

# SMOTE후
# loss :  0.3335728645324707
# acc :  0.8928571343421936
# acc :  0.8928571428571429
# f1 :  0.6188505747126437


























