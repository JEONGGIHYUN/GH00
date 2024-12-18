from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
import tensorflow as tf
tf.random.set_seed(337)
np.random.seed(337)

datasets = load_boston()
x = datasets.data
y = datasets.target

from sklearn.preprocessing import RobustScaler, StandardScaler, Normalizer

scaler = StandardScaler()
x = scaler.fit_transform(x)


from sklearn.model_selection import train_test_split

x_train, x_test ,y_train, y_test = train_test_split(x, y, train_size=0.8,random_state=135)

model = Sequential()
model.add(Dense(10, input_dim=13))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일 훈련
from tensorflow.keras.optimizers import Adam, Adagrad
from tensorflow.keras.losses import mse, mae
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau

es = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1, restore_best_weights=True)

rl = ReduceLROnPlateau(monitor='val_loss',mode='auto',patience=10, verbose=1,factor=0.9)

for i in range(6): 
    learning_rate = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    # learning_rate = 0.0007       # default = 0.001
    learning_rate = learning_rate[i]
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate[i]))

# learning_rate = 0.001 #default
# learning_rate = 0.01

# model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

    model.fit(x_train, y_train, validation_split=0.2, epochs=100, batch_size=32,verbose=0,callbacks=[es,rl])

#4 평가 예측
#4. 평가,예측
    print("=================1. 기본출력 ========================")
    loss = model.evaluate(x_test, y_test, verbose=0)
    print('lr : {0}, 로스 :{1}'.format(learning_rate, loss))

    y_predict = model.predict(x_test, verbose=0)
    r2 = r2_score(y_test, y_predict)
    print('lr : {0}, r2 : {1}'.format(learning_rate, r2))