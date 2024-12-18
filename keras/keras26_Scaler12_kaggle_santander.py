# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
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
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)
###############################################
# print(x_train)
# print(np.min(x_train), np.max(x_train)) #
# print(np.min(x_test), np.max(x_test)) # 

#2. 모델 구성
model = Sequential()
model.add(Dense(500, activation='relu', input_dim=200))
model.add(Dense(450, activation='relu'))
model.add(Dense(400))
model.add(Dense(350, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
start_time = time.time()
es = EarlyStopping(
    monitor='val_loss',
    mode = 'min',
    patience=10,
    restore_best_weights=True
)

model.fit(x_train, y_train, epochs=101, batch_size=4096, validation_split=0.2)

end_time = time.time()

#4. 예측 평가
loss = model.evaluate(x_test, y_test)
y_predict = np.around(model.predict(x_test))
y_submit = model.predict(test_csv)
accuracy_score = accuracy_score(y_test, y_predict)

submission_csv['target'] =(y_submit)

print('loss :', loss)
print('acc score :', accuracy_score)
print('time :', round(end_time - start_time, 2), '초')

submission_csv.to_csv(path + 'submission_0724_16_12.csv')
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


'''