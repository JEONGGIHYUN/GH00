import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, concatenate, Concatenate, BatchNormalization, Reshape, Conv1D, Input, LSTM, Bidirectional
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
import time, datetime
from sklearn.model_selection import train_test_split

naver = pd.read_csv('C:/TDS/ai5/_data/성우하이텍/NAVER 240816(2).csv', index_col=[0,6], thousands=',')

hybe = pd.read_csv('C:/TDS/ai5/_data/성우하이텍/하이브 240816(1).csv', index_col=[0,6], thousands=',')

swh = pd.read_csv('C:/TDS/ai5/_data/성우하이텍/성우하이텍 240816(1).csv', index_col=[0,6], thousands=',')

# naver[['시가','고가','저가','종가','거래량','금액(백만)','개인','기관','외인(수량)','외국계','프로그램']] = naver[['시가','고가','저가','종가','거래량','금액(백만)','개인','기관','외인(수량)','외국계','프로그램']].astype('float')

# hybe[['시가','고가','저가','종가','거래량','금액(백만)','개인','기관','외인(수량)','외국계','프로그램']] = hybe[['시가','고가','저가','종가','거래량','금액(백만)','개인','기관','외인(수량)','외국계','프로그램']].astype('float')

# swh[['시가','고가','저가','종가','거래량','금액(백만)','개인','기관','외인(수량)','외국계','프로그램']] = swh[['시가','고가','저가','종가','거래량','금액(백만)','개인','기관','외인(수량)','외국계','프로그램']].astype('float')

naver = naver.sort_values(by=['일자'])
hybe = hybe.sort_values(by=['일자'])
swh = swh.sort_values(by=['일자'])

###########################################################################
x1 = naver.drop(['전일비', '프로그램', '외국계', '외인(수량)', '기관', '거래량'], axis=1)

x2 = hybe.drop(['전일비', '프로그램', '외국계', '외인(수량)', '기관', '거래량'], axis=1)

y = swh.drop(['전일비', '프로그램', '외국계', '외인(수량)', '기관', '거래량'], axis=1)

x1 = x1.tail(948)
x2 = x2.tail(948)
y1 = x1.tail(20)
y2 = x2.tail(20)
y = swh.tail(948)

print(x1, x2, y1, y2, y)


y = y['종가']
print(y.shape)

x1 = x1.to_numpy()
x2 = x2.to_numpy()
y1 = y1.to_numpy()
y2 = y2.to_numpy()
y = y.to_numpy()



size = 20

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

x1 = split_x(x1, size)
print(x1.shape) # (929, 8, 6)
x2 = split_x(x2, size)
print(x2.shape) # (929, 8, 6)

y = split_x(y, size)
print(y.shape) # (929, 8)


x1_pre = y1.reshape(1, 20, 9)

x2_pre = y2.reshape(1, 20, 9)

y = y.reshape(929, 20, 1)



# x1_train, x1_test, x2_train, x2_test, x3_train, x3_test = train_test_split(x1, x2, x3, train_size=0.8, random_state=4)

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, train_size=0.85, random_state=4)

print(x1.shape, x2.shape, y.shape)



# print(x1_train.shape, x2_train.shape, x3_train.shape)
# (3874, 30, 13) (750, 10, 13) (2783, 20, 13)
# 2-1. 함수 모델 구성
input1 = Input(shape=(20, 9))
dense1_1 = LSTM(64, input_shape=(1,1),return_sequences=True)(input1)
dense1_2 = LSTM(32, input_shape=(2,2))(dense1_1)
dense1_3 = Reshape(target_shape=(8,4))(dense1_2)
dense1_4 = Conv1D(filters=32, kernel_size=2, activation='relu')(dense1_3)


#2-2
input2 = Input(shape=(20, 9))
dense2_1 = LSTM(64, input_shape=(1,1),return_sequences=True)(input2)
dense2_2 = LSTM(32, input_shape=(2,2))(dense2_1)
dense2_3 = Reshape(target_shape=(8,4))(dense2_2)
dense2_4 = Conv1D(filters=32, kernel_size=2, activation='relu')(dense2_3)



# # 2-1. 함수 모델 구성
# input1 = Input(shape=(8, 11))
# # dense2_1 = Reshape(target_shape=(30,13))
# dense1_1 = Conv1D(filters=32 , kernel_size=2)(input1)
# dense1_2 = Conv1D(filters=16 , kernel_size=2)(dense1_1)
# dense1_3 = LSTM(16)(dense1_2)

# #2-2
# input2 = Input(shape=(8, 11))
# dense2_1 = Conv1D(filters=32, kernel_size=2)(input2)
# dense2_2 = Conv1D(filters=16 , kernel_size=2)(dense2_1)
# dense2_3 = LSTM(16)(dense2_2)

# 2-4 합하기
merge1 = Concatenate()([dense1_4,dense2_4])
dense2_1 = Conv1D(10, kernel_size=1, activation='relu')(merge1)
dense2_2 = Flatten()(dense2_1)
dense2_3 = Dense(300, activation='relu')(dense2_2)
dense2_4 = Dense(200, activation='relu')(dense2_3)
dense2_5 = Dense(100, activation='relu')(dense2_4)
dense2_6 = Dense(50, activation='relu')(dense2_5)
output1 = Dense(1)(dense2_6)

model = Model(inputs=[input1,input2], outputs=output1)

model.summary()


#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam',)

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    verbose=1,
    restore_best_weights=True
)

date = datetime.datetime.now()

date = date.strftime("%m%d.%H%M")

path = 'C:/TDS/ai5/_save/성우하이텍/'
filename = '{epoch:04d}_valloss_{val_loss:.4f}.hdf5'
filepath = "".join([path, '성우하이텍', date, '__epo__', filename])

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

start = time.time()
model.fit([x1_train, x2_train], y_train,
          epochs=1000,
          batch_size=64
          ,
          validation_split=0.2,
          callbacks=[es, mcp])
end = time.time()

#4. 평가 예측
loss = model.evaluate([x1_test,x2_test],y_test)

y_pred = model.predict([x1_pre,x2_pre])
print('소요시간 :', round(end - start,2), '초')
print('로스 :', loss)
print('8월 19일 월요일 종가의 예측 결과:', y_pred)

"""
소요시간 : 46.21 초
로스 : 3953792.0
8월 19일 월요일 종가의 예측 결과: [[6400.042]] 배치 1024


"""





























































