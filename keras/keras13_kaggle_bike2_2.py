import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

path = 'C:/TDS/ai5/_data/bike-sharing-demand/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test_01.csv', index_col=0)
sampleSubmission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)

x = train_csv.drop(['count'], axis=1)

y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.87, random_state=57543)

model = Sequential()
model.add(Dense(16, activation='relu', input_dim=10))
model.add(Dense(256, activation='relu'))
model.add(Dense(512))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=300, batch_size=1000)

loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)

y_submit = model.predict(test_csv)

sampleSubmission['count'] = y_submit

r2 = r2_score(y_test, results)
print('로스 :', loss)
print('r2 :', r2)
sampleSubmission.to_csv(path + 'submission_0718_1209.csv')