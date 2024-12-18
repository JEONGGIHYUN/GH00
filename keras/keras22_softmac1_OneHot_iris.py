import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import time


#1. 데이터
datasets = load_iris()
# print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data # (150, 4)
y = datasets.target # (150,)
# print(x.shape,  y.shape)
# print(y)

# print(np.unique(y, return_counts=True))
# (array([0, 1, 2]), array([50, 50, 50], dtype=int64))

# print(pd.value_counts(y))
# (array([0, 1, 2]), array([50, 50, 50], dtype=int64))
# 0    50
# 1    50
# 2    50
# dtype: int64

######################################################
# 판다스
y = pd.get_dummies(y)
print(y)
print(y.shape)

# 케라스
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# print(y)
# print(y.shape)

#리쉐이프의 조건은 데이터의 값이 바뀌면 안된다. 데이터의 순서가 바뀌면 안된다.
# 사이킷런
# from sklearn.preprocessing import OneHotEncoder
# y_ohe3 = y.reshape(-1, 1)
# y_ohe = OneHotEncoder(sparse=False) #True가 기본값
# y_ohe3 = y_ohe.fit_transform(y_ohe3)
# print(y_ohe3)



######################################################

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=2356, train_size=0.8, shuffle=True, stratify=y)


#2. 모델구성
model = Sequential()
model.add(Dense(500, input_dim=4, activation='relu'))
model.add(Dense(250))
model.add(Dense(125))
model.add(Dense(75))
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(10))
model.add(Dense(3, activation='softmax'))

#3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
start_time = time.time()
es = EarlyStopping(
    monitor='val_loss',
    mode = 'min',
    patience=10,
    restore_best_weights=True
)

model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

end_time = time.time()

#4. 예측 평가
loss = model.evaluate(x_test, y_test)
y_predict = np.around(model.predict(x_test))
# accuracy_score = accuracy_score(y_test, y_predict)
results = np.around(model.predict(x))
# print('acc score :', accuracy_score)
print('loss :', loss)
print('time :', round(end_time - start_time, 2), '초')
print('results :', results)















