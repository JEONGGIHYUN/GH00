import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from keras.layers import Dense, Input
from sklearn.metrics import r2_score
from keras.layers.merge import Concatenate, concatenate
from keras.callbacks import EarlyStopping

#1. 데이터
x1_datasets = np.array([range(100), range(301, 401)]).T
                       # 삼성 종가, 하이닉스 종가
x2_datasets = np.array([range(101, 201), range(411, 511),
                              range(150, 250)]).transpose()
                       # 원유, 환율, 금시세
x3_datasets = np.array([range(100), range(301,401),
                        range(77,177),range(33,133)]).T
# 가나다라마바사아자차카타파하
y = np.array(range(3001, 3101)) # 한강의 화씨 온도.

# x1_train, x1_test, y1_train, y1_test = train_test_split(x1_datasets, y, random_state=4, train_size=0.7)

# x2_train, x2_test, y2_train, y2_test = train_test_split(x2_datasets, y, random_state=4, train_size=0.7)

x1_train, x1_test, x2_train, x2_test,x3_train,x3_test, y_train, y_test = train_test_split(
    x1_datasets, x2_datasets,x3_datasets, y, train_size=0.7, random_state=4,
)

print(x1_train.shape, x2_train.shape, x3_train.shape, y_train.shape) # (70, 2) (70, 3) (70, 4) (70,)

#2-1. 함수형 모델
input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu', name='bit1')(input1)
dense2 = Dense(20, activation='relu', name='bit2')(dense1)
dense3 = Dense(30, activation='relu', name='bit3')(dense2)
dense4 = Dense(40, activation='relu', name='bit4')(dense3)
output1 = Dense(50, activation='relu', name='bit5')(dense4)
# model1 = Model(inputs=input1, outputs=output1)

input2 = Input(shape=(3,))
dense11 = Dense(100, activation='relu', name='bit11')(input2)
dense22 = Dense(200, activation='relu', name='bit22')(dense11)
output2 = Dense(300, activation='relu', name='bit33')(dense22)
# model2 = Model(inputs=input2, outputs=output2)

input3 = Input(shape=(4,))
dense111 = Dense(100, activation='relu', name='bit111')(input3)
dense222 = Dense(200, activation='relu', name='bit222')(dense111)
output3 = Dense(300, activation='relu', name='bit333')(dense222)

#2-3 합하기
merge1 = Concatenate(name='mg1')([output1,output2,output3])
merge2 = Dense(7, name='mg2')(merge1)
merge3 = Dense(20, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs=[input1,input2,input3],outputs=last_output)
# model.summary()

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=100, verbose=1,
                   restore_best_weights=True)

model.fit([x1_train, x2_train, x3_train], y_train, 
          epochs=1000,
          verbose=2,
          callbacks=[es]
        #   validation_split=0.3
          )

#4. 평가 예측
loss1 = model.evaluate([x1_test,x2_test,x3_test],y_test)
# results1 = model.predict([x1_test,x2_test])
x1_pred = np.array([range(101,110), range(401, 410)]).T
                       # 삼성 종가, 하이닉스 종가
x2_pred = np.array([range(201, 210), range(511, 520),
                              range(250, 259)]).transpose()
x3_pred = np.array([range(101,110), range(401,410),
                        range(178,187),range(134,143)]).T
results2 = model.predict([x1_pred,x2_pred,x3_pred])
# r2 = r2_score(y_test, results2)
print('로스 :', loss1)
print(' :', results2)
# print('소요시간 :', round(end_time - start_time), '초')
