# https://dacon.io/competitions/official/236068/overview/description

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
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

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

#2. 모델 구성
model = Sequential()
model.add(Dense(50, input_dim=8))
model.add(Dense(100, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일 훈련
start = time.time()

model.compile(loss='mse', optimizer='adam', metrics='acc') # accuracy, mse

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience=50,
    restore_best_weights=True
)

hist = model.fit(x_train, y_train, epochs=1000,
                 verbose=1,
                 validation_split=0.4,
                 callbacks=[es])

end = time.time()


#4. 평가 예측
loss = model.evaluate(x_test, y_test)
y_pred = np.round(model.predict(x_test))
y_submit = model.predict(test_csv)
print(y_pred)
acc = accuracy_score(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

submission_csv['Outcome'] = np.round(y_submit)
print('로스 :', loss)

submission_csv.to_csv(path + 'submission_0722_1635.csv')

# y_pred = np.round(y_pred)

# print(y_pred)

# accuracy_score = accuracy_score(y_test, y_pred)








