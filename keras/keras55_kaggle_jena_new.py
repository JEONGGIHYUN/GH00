# 메일제목 : jena 홍길동
# 내용
# 첨부 jena_홍길동.py
# jena_홍길동.h5
# or
# jena_홍길동.hdf5
# _data//_save//keras55

import pandas as pd
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, LSTM, SimpleRNN, GRU
import time
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
import os
path1 = "C:/TDS/ai5/_data/jena/"

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
data = pd.read_csv(path1 + "cs2.csv",index_col=0)

x = data.head(420407)
y = x['T (degC)']
x = x.drop(["T (degC)"], axis =1)
y_pre = data.tail(144)["T (degC)"]

size = 144
def split_x(data, size):
    aaa=[]
    for i in range(len(data) - size + 1):
        sub = data[i : (i+size)]
        aaa.append(sub)
    return np.array(aaa)

x = split_x(x, size)
y = split_x(y, size)

x_test1 = x[-1].reshape(-1,144,18)
# print(x.shape)
# print(y.shape)
x = np.delete(x, -1, axis =0)
y = np.delete(y, 0, axis = 0)
# print(x.shape)
# print(y.shape)
# y_pre = split_x(y_pre ,size)

# y_pre = np.delete(y_pre, 1, axis =1)

print(y_pre.shape) # (1, 143)




x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

"""
(336210, 144, 18)
(84053, 144, 18)
(336210, 144)
(84053, 144)
"""

print(x_train.shape)
print(x_test.shape)


#2. 모델 구성
model = Sequential()
model.add(LSTM(50, input_shape = (144, 18)))
model.add(Dense(144))

model.add(Dense(144))
model.add(Dense(144))
model.add(Dense(144))

#3. 컴파일 및 훈련
from keras.callbacks import EarlyStopping, ModelCheckpoint
model.compile(loss = 'mse', optimizer='adam')

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience=30,
    restore_best_weights=True
)

# mcp = ModelCheckpoint(
    # monitor='val_loss',
    # mode = 'auto',
    # verbose=1,
#     save_best_only=True,
#     filepath="C:\\ai5\\_save\\keras55\\keras55_02_.hdf5"
# )

model.fit(x_train, y_train,
          epochs=100,
          batch_size=1024,
          validation_split=0.2,
          callbacks=[es])#,mcp])

#4. 예측 및 평가
loss = model.evaluate(x_test,y_test)
result = model.predict(x_test1)
result = np.array([result]).reshape(144,1)
# acc = accuracy_score(y_pre, result)

print(loss, result)
# print(acc)
print(result.shape)
# print(y_pre)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_pre, result)

print(rmse)

# 3.3620394403080707