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

# 가나다라마바사아자차카타파하
y1 = np.array(range(3001, 3101)) # 한강의 화씨 온도.
y2 = np.array(range(13001, 13101))

# x1_train, x1_test, y1_train, y1_test = train_test_split(x1_datasets, y, random_state=4, train_size=0.7)

# x2_train, x2_test, y2_train, y2_test = train_test_split(x2_datasets, y, random_state=4, train_size=0.7)

x_train, x_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1_datasets, y1, y2, train_size=0.7, random_state=4,
)

print(x_train.shape, y1_train.shape, y2_train.shape) # (70, 2) (70, 3) (70, 4) (70,)

#2-1. 함수형 모델
input1 = Input(shape=(2,))
dense1 = Dense(100, activation='relu', name='bit1')(input1)
dense2 = Dense(200, activation='relu', name='bit2')(dense1)
dense3 = Dense(300, activation='relu', name='bit3')(dense2)
dense4 = Dense(400, activation='relu', name='bit4')(dense3)
dense5 = Dense(500, activation='relu', name='bit5')(dense4)
# model1 = Model(inputs=input1, outputs=output1)

#2-3 합하기
merge1 = Concatenate(name='mg1')([dense5])
merge2 = Dense(100, name='mg2')(merge1)
merge3 = Dense(200, name='mg3')(merge2)
merge4 = Dense(300, name='last')(merge3)

output1_1 = Dense(1)(merge4)

output2_1 = Dense(1)(merge4)


model = Model(inputs=[input1],outputs=[output1_1,output2_1])
# model.summary()

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='loss', mode='min',
                   patience=100, verbose=1,
                   restore_best_weights=True)

model.fit(x_train, [y1_train,y2_train], 
          epochs=1000,
          verbose=2,
          callbacks=[es]
        #   validation_split=0.3
          )

#4. 평가 예측
loss1 = model.evaluate(x_test,[y1_test,y2_test])
# results1 = model.predict([x1_test,x2_test])
x_pred = np.array([range(101,110), range(401, 410)]).T
                       # 삼성 종가, 하이닉스 종가

results1, results2 = model.predict(x_pred)
# r2 = r2_score(y_test, results2)
print('로스 :', loss1)
print('results', results1)
print(' :', results2)
# print('소요시간 :', round(end_time - start_time), '초')
