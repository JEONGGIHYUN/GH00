import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

#1. 데이터
datasets = fetch_covtype()
# print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data # (581012, 54)
y = datasets.target # (581012,)
# print(x.shape, y.shape)

# print(np.unique(y, return_counts=True))
#  (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],
    #   dtype=int64))

# print(pd.value_counts(y))
# 2    283301
# 1    211840
# 3     35754
# 7     20510
# 6     17367
# 5      9493
# 4      2747
# dtype: int64

# y = pd.get_dummies(y)
# print(y)
# print(y.shape)

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape)

# from sklearn.preprocessing import OneHotEncoder
# y_ohe3 = y.reshape(-1, 1)
# y_ohe = OneHotEncoder(sparse=False) #True가 기본값
# y_ohe3 = y_ohe.fit_transform(y_ohe3)
# print(y_ohe3)
# print(y_ohe3.shape)


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1356, train_size=0.75, shuffle=True, stratify=y)

#2. 모델 구성
model = Sequential ()
model.add(Dense(500, input_dim=54, activation='relu'))
model.add(Dense(250))
model.add(Dense(125))
model.add(Dense(75))
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(10))
model.add(Dense(8, activation='softmax'))

#3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
start_time = time.time()
es = EarlyStopping(
    monitor='val_loss',
    mode = 'min',
    patience=10,
    restore_best_weights=True
)

model.fit(x_train, y_train, epochs=30, batch_size=10000, validation_split=0.2)

end_time = time.time()

#4. 예측 평가
loss = model.evaluate(x_test, y_test)
y_predict = np.around(model.predict(x_test))
accuracy_score = accuracy_score(y_test, y_predict)

print('loss :', loss)
print('acc score :', accuracy_score)
print('time :', round(end_time - start_time, 2), '초')

# loss : [0.4637472927570343, 0.8029851317405701]
# acc score : 0.7942693094118538
# time : 108.65 초











