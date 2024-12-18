import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense,SimpleRNN,LSTM,GRU
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

save_csv = 'C:/TDS/ai5/_data/LSTM_scale save/'

#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50,],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

# x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
#               [5,6,7],[6,7,8],[7,8,9],[8,9,10],
#               [9,10,11],[10,11,12],
#               [20,30,40],[30,40,50],[40,50,60]])
# y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x = x.reshape(x.shape[0],x.shape[1], 1)

# print(x.shape) (7, 3, 1)
# 3-D tensor with shape (batch_size, timesteps, features*)

# #2 모델구성
model = Sequential()
model.add(SimpleRNN(units=21, input_shape=(3,1), activation='relu',return_sequences=True)) 
model.add(SimpleRNN(units=21, activation='relu'))
model.add(Dense(20)) # RNN은 Dense와 바로 연결이 가능하다.
model.add(Dense(15, activation='relu'))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
es = EarlyStopping(
    monitor='loss',
    mode='min',
    patience=30,
    verbose=1,
    restore_best_weights=True
)
import datetime

date = datetime.datetime.now()

date = date.strftime('%m%d_%H%M')

filename = '{epoch:04d}-{loss:.4f}.hdf5'
filepath = ''.join([save_csv, 'lstm', date,'_', filename])

mcp = ModelCheckpoint(
    monitor='loss',
    mode = 'auto',
    verbose= 1,
    save_best_only=True,
    filepath = filepath
)


model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000,
          callbacks=[es, mcp],
        #   validation_split=0.1
          )

#4. 평가, 예측
# print('================== 1. save.model 출력====================')
# model = load_model('C:/TDS/ai5/_data/LSTM_scale save/79.99879.hdf5')

results = model.evaluate(x, y)
print('loss :', results)

x_pred = x_pred = np.array([50,60,70]).reshape(1, 3, 1) # [[8],[9],[10]]
y_pred = model.predict(x_pred)
# y_pred = np.round(y_pred,1)

print('[50,60,70]의 결과 :', y_pred)

# loss : 0.0013975021429359913
# [50,60,70]의 결과 : [[79.98414]]

# loss : 0.0006566010997630656
# [50,60,70]의 결과 : [[81.239944]]

# loss : 13.229738235473633
# [50,60,70]의 결과 : [[94.410736]]

# loss : 0.006571746896952391
# [50,60,70]의 결과 : [[80.20533]]

# loss : 0.0008009340381249785
# [50,60,70]의 결과 : [[79.99317]]





