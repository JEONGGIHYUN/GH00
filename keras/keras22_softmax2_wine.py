from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import time

#1. 데이터
datasets = load_wine()
# print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data # (178, 13)
y = datasets.target # (178,)
# print(x.shape, y.shape)

# print(np.unique(y, return_counts=True))
# (array([0, 1, 2]), array([59, 71, 48], dtype=int64))

# print(pd.value_counts(y))
# (array([0, 1, 2]), array([59, 71, 48], dtype=int64))
# 1    71
# 0    59
# 2    48

y = pd.get_dummies(y)
print(y)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=135, train_size=0.7, shuffle=True, stratify=y)

#2. 모델 구성
model = Sequential()
model.add(Dense(500, input_dim=13, activation='relu'))
model.add(Dense(231, activation='relu'))
model.add(Dense(131, activation='relu'))
model.add(Dense(31, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
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

model.fit(x_train, y_train, epochs=1000, batch_size=100, validation_split=0.1)

end_time = time.time()

#4. 예측 평가
loss = model.evaluate(x_test, y_test)

y_predict = np.around(model.predict(x_test))

accuracy_score = accuracy_score(y_test, y_predict)

print('loss :', loss)
print('time :', round(end_time - start_time, 2), '초')
print('acc score :', accuracy_score)


