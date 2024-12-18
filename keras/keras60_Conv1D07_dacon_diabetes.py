# https://dacon.io/competitions/official/236068/overview/description

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten, Conv1D, Reshape
from tensorflow.keras.callbacks import EarlyStopping
import time


#1. 데이터
path = 'C:/TDS/ai5/_data/dacon/diabetes/'

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

# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)

################################################
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
scaler = RobustScaler()

scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)
################################################
# # print(x_train)
# print(np.min(x_train), np.max(x_train)) # 
# print(np.min(x_test), np.max(x_test)) #

#2. 모델 구성
model = Sequential()
model.add(Reshape(target_shape=(4,2)))
model.add(Conv1D(filters=10,kernel_size=2,input_shape=(4,2)))
model.add(Flatten())
model.add(Dense(50))
model.add(Dense(100, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# #2. 함수형 모델 구성
# input1 = Input(shape=(8, ))
# dense1 = Dense(50)(input1)
# dense2 = Dense(100, activation='relu')(dense1)
# dense3 = Dense(150, activation='relu')(dense2)
# dense4 = Dense(200, activation='relu')(dense3)
# dense5 = Dropout(0.2)(dense4)
# dense6 = Dense(100, activation='relu')(dense5)
# dense7 = Dense(50, activation='relu')(dense6)
# dense8 = Dense(20, activation='relu')(dense7)
# output1 = Dense(1, activation='sigmoid')(dense8)
# model = Model(inputs=input1, outputs=output1)

#3. 컴파일 훈련

model.compile(loss='mse', optimizer='adam', metrics='acc') # accuracy, mse

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
filepath = ''.join([path, 'k32_07_', date, '_', filename])
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
model.fit(x_train, y_train, epochs=3000,  batch_size=32,
                 verbose=2,
                 validation_split=0.3,
                 callbacks=[es, mcp]
                 )

end_time = time.time()

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
y_pred = np.round(model.predict(x_test))
y_submit = model.predict(test_csv)
print(y_pred)
acc = accuracy_score(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# submission_csv['Outcome'] = np.round(y_submit)
print('로스 :', loss)
print('소요시간 :', round(end_time - start_time), '초')

# submission_csv.to_csv(path + 'submission_0722_1635.csv')

# 로스 : [0.2036416232585907, 0.6989796161651611]

# 로스 : [0.20923519134521484, 0.6887755393981934]

# 로스 : [0.21842415630817413, 0.6938775777816772]

# MaxAbsScaler
# 로스 : [0.30629029870033264, 0.6224489808082581]

# 로스 : [0.2141127735376358, 0.6836734414100647]

# RobustScaler
# 로스 : [0.3571428656578064, 0.6428571343421936]

# 로스 : [0.21126089990139008, 0.6938775777816772]
# -----------------------------------------------------
# 로스 : [0.3430812358856201, 0.6122449040412903]

# cpu 소요시간 : 1 초

# gpu 소요시간 : 3 초

#Conv1D
# 로스 : [0.2083785980939865, 0.6989796161651611]
# 소요시간 : 5 초

