# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Reshape, Conv1D, Flatten
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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
scaler = RobustScaler()

scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)
###############################################
# print(x_train)
# print(np.min(x_train), np.max(x_train)) #
# print(np.min(x_test), np.max(x_test)) # 

#2. 모델 구성
model = Sequential()
model.add(Reshape(target_shape=(50,4)))
model.add(Conv1D(filters=10,kernel_size=2,input_shape=(50,4)))
model.add(Flatten())
model.add(Dense(500))
model.add(Dense(450, activation='relu'))
model.add(Dense(400))
model.add(Dense(350, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(200, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# #2. 함수형 모델 구성
# input1 = Input(shape=(200, ))
# dense1 = Dense(500, activation='relu')(input1)
# dense2 = Dense(450, activation='relu')(dense1)
# dense3 = Dense(400)(dense2)
# dense4 = Dense(350, activation='relu')(dense3)
# dense5 = Dense(300, activation='relu')(dense4)
# dense6 = Dropout(0.3)(dense5)
# dense7 = Dense(200, activation='relu')(dense6)
# dense8 = Dense(150, activation='relu')(dense7)
# dense9 = Dense(100, activation='relu')(dense8)
# dense10 = Dense(50, activation='relu')(dense9)
# output1 = Dense(1, activation='sigmoid')(dense10)
# model = Model(inputs=input1, outputs=output1)

#3. 컴파일 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

start_time = time.time()

################
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
################
es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=10, verbose=1,
                   restore_best_weights=True)
################
################파일 명 만 들 기
import datetime
date = datetime.datetime.now()
print(date)
print(type(date)) # <class 'datetime.datetime'>
date = date.strftime('%m%d_%H%M')
print(date)
print(type(date))

path = 'C:/TDS/ai5/study/_save/keras32_/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path, 'k32_12_', date, '_', filename])
################
################
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode = 'auto',
    verbose= 1,
    save_best_only=True,
    filepath = filepath
)
################
model.fit(x_train, y_train, epochs=3000,  batch_size=4096,
                 verbose=2,
                 validation_split=0.3,
                 callbacks=[es, mcp]
                 )

end_time = time.time()

#4. 예측 평가
loss = model.evaluate(x_test, y_test)
y_predict = np.around(model.predict(x_test))
y_submit = model.predict(test_csv)
accuracy_score = accuracy_score(y_test, y_predict)

submission_csv['target'] =(y_submit)

print('loss :', loss)
print('acc score :', accuracy_score)
print('소요시간 :', round(end_time - start_time), '초')

# submission_csv.to_csv(path + 'submission_0725_16_51.csv')
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

loss : [1.214404821395874, 0.8827499747276306]
acc score : 0.88275
time : 140.57 초

MaxAbsScaler
loss : [1.0676288604736328, 0.885016679763794]
acc score : 0.8850166666666667
time : 135.4 초

loss : [1.1891380548477173, 0.8880166411399841]
acc score : 0.8880166666666667
time : 100.43 초

RobustScaler
loss : [1.0957788228988647, 0.8907666802406311]
acc score : 0.8907666666666667
time : 67.54 초

loss : [1.0016027688980103, 0.8838333487510681]
acc score : 0.8838333333333334
time : 68.88 초

================================================
loss : [0.24954678118228912, 0.9097333550453186]
acc score : 0.9097333333333333


# cpu 소요시간 : 24 초

# gpu 소요시간 : 5 초
'''

# Conv1D
# loss : [0.24759341776371002, 0.9090666770935059]
# acc score : 0.9090666666666667
# 소요시간 : 11 초