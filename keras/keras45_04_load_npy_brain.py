# 배치를 160잡고
# x, y를 추출해서 모델을 만들기
# acc 0.99 이상
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D,Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator


np_path = 'C:/TDS/ai5/_data/_save_npy/'
x_train = np.load(np_path + 'keras45_01_x_train.npy')

y_train = np.load(np_path + 'keras45_01_y_train.npy')

x_test = np.load(np_path + 'keras45_01_x_test.npy')

y_test = np.load(np_path + 'keras45_01_y_test.npy')


#2. 모델 구성
model = Sequential()
model.add(Conv2D(30,(2,2), input_shape=(200,200,1),padding='same'))
model.add(Conv2D(20, (2,2), activation='relu',padding='same'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) # accuracy, mse

################
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
################
es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=10, verbose=1,
                   restore_best_weights=True)
################

model.fit(x_train, y_train, epochs=30,  batch_size=1000,
                 verbose=2,
                 validation_split=0.3,
                 callbacks=[es]
                 )

#4. 평가 예측
loss = model.evaluate(x_test, y_test)


y_pred = model.predict(x_test)
print(y_pred)

y_pred = np.round(y_pred)

accuracy_score = accuracy_score(y_test, y_pred)


print('로스 :', loss)
print('acc_score :', accuracy_score)

