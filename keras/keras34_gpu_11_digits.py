from sklearn.datasets import load_digits
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

x, y = load_digits(return_X_y=True) # 사이킷런에서 사용 가능한 방식이다.

# print(x)
# print(y)
# print(x.shape, y.shape) #(1797, 64) (1797,)

print(pd.value_counts(y, sort=False))
# 0    178
# 1    182
# 2    177
# 3    183
# 4    181
# 5    182
# 6    181
# 7    179
# 8    174
# 9    180
# dtype: int64

y = pd.get_dummies(y)
print(y)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1346, train_size=0.75, shuffle=True, stratify=y)

###############################################
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
scaler = RobustScaler()

scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)
################################################
# print(x_train)
print(np.min(x_train), np.max(x_train)) # 0.0 16.0
print(np.min(x_test), np.max(x_test)) # 0.0 16.0


# #2. 모델 구성
# model = Sequential ()
# model.add(Dense(500, input_dim=64, activation='relu'))
# model.add(Dense(250))
# model.add(Dense(125))
# model.add(Dropout(0.3))
# model.add(Dense(50))
# model.add(Dense(25))
# model.add(Dense(10))
# model.add(Dense(10, activation='softmax'))

#2. 함수형 모델 구성
input1 = Input(shape=(64, ))
dense1 = Dense(500, activation='relu')(input1)
dense2 = Dense(250)(dense1)
dense3 = Dense(125)(dense2)
dense4 = Dropout(0.3)(dense3)
dense5 = Dense(50)(dense4)
dense6 = Dense(25)(dense5)
dense7 = Dense(10)(dense6)
output1 = Dense(10, activation='softmax')(dense7)
model = Model(inputs=input1, outputs=output1)

#3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

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
filepath = ''.join([path, 'k32_11_', date, '_', filename])
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
model.fit(x_train, y_train, epochs=3000,  batch_size=10000,
                 verbose=2,
                 validation_split=0.3,
                 callbacks=[es, mcp]
                 )

end_time = time.time()

#4. 예측 평가
loss = model.evaluate(x_test, y_test)
y_predict = np.around(model.predict(x_test))
accuracy_score = accuracy_score(y_test, y_predict)

print('loss :', loss)
print('acc score :', accuracy_score)
print('소요시간 :', round(end_time - start_time), '초')

'''
loss : [0.16231749951839447, 0.9733333587646484]
acc score : 0.9733333333333334
time : 49.37 초

loss : [0.07097781449556351, 0.9777777791023254]
acc score : 0.9777777777777777
time : 49.53 초

loss : [0.1270405799150467, 0.9755555391311646]
acc score : 0.9755555555555555
time : 49.08 초

MaxAbsScaler
loss : [0.13909640908241272, 0.9733333587646484]
acc score : 0.9733333333333334
time : 49.13 초

loss : [0.24305672943592072, 0.9688888788223267]
acc score : 0.9688888888888889
time : 54.53 초

RobustScaler
loss : [0.21911989152431488, 0.9644444584846497]
acc score : 0.9644444444444444
time : 49.56 초

loss : [0.2355182021856308, 0.9733333587646484]
acc score : 0.9711111111111111
time : 49.4 초

====================================================
loss : [0.10504543781280518, 0.9711111187934875]
acc score : 0.9688888888888889

# cpu 소요시간 : 3 초

# gpu 소요시간 : 3 초
'''