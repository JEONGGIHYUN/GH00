import numpy as np
from sklearn.datasets import load_breast_cancer
import pandas as pd
from tensorflow.keras.models import Sequential
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


#2. 모델 구성
model = Sequential()
model.add(Dense(40, input_dim=30))
model.add(Dense(100, activation='relu'))
model.add(Dense(125, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(150, activation='sigmoid'))
model.add(Dense(70, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일 훈련
start = time.time()

model.compile(loss='mse', optimizer='adam', metrics=['acc']) # accuracy, mse

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience=20,
    restore_best_weights=True
)

hist = model.fit(x_train, y_train, epochs=1000,
          verbose=1,
          validation_split=0.3,
          callbacks=(es))



end = time.time()

#4. 평가 예측
loss = model.evaluate(x_test, y_test)


y_pred = model.predict(x_test)
print(y_pred)
r2 = r2_score(y_test, y_pred)
y_pred = np.round(y_pred)

print(y_pred)

accuracy_score = accuracy_score(y_test, y_pred)


print('로스 :', loss)
print('acc_score :', accuracy_score)
print('걸린 시간 :', round(end - start, 2), '초')
print('r2 스코어 :', r2)

# print('소요시간 :', round(end_time - start_time),'초')












