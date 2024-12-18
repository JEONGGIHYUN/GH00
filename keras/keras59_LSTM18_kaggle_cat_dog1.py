# https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/overview

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D,Flatten, Dropout, Reshape, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,    # 수평 뒤집기
    vertical_flip=True,      # 수직 뒤집기
    width_shift_range=0.4,   # 평행 이동
    height_shift_range=0.2,  # 평행 이동 수직
    rotation_range=6,        # 정해진 각도만큼 이미지 회전
    zoom_range=1.3,          # 축소 또는 확대
    shear_range=0.7,         # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환
    fill_mode='nearest',     # 데이터의 비어있는 곳을 가까운 데이터와 비슷한 값으로 채움 
)


test_datagen = ImageDataGenerator(
    rescale= 1./255
)
# start_time = time.time()
path_train = 'C:/TDS/ai5/_data/kaggle/dogs-vs-cats-redux-kernels-edition/train/train/'
path_test = 'C:/TDS/ai5/_data/kaggle/dogs-vs-cats-redux-kernels-edition/test/'
path_csv = 'C:/TDS/ai5/_data/kaggle/dogs-vs-cats-redux-kernels-edition/'

xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(110,110),
    batch_size=25000,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True,
)

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(110,110),
    batch_size=12500,
    class_mode='binary',
    color_mode='rgb',
)

submission_csv = pd.read_csv(path_csv + 'sample_submission.csv', index_col=0)

x_train, x_test, y_train, y_test = train_test_split(xy_train[0][0], xy_train[0][1], train_size=0.8, random_state=42)

# x_train = xy_train[0][0]
# y_train = xy_train[0][1]
# x_test = xy_test[0][0]
# y_test = xy_test[0][1]

# end_time = time.time()

print(x_train.shape,x_test.shape) # (14000, 100, 100, 3) (6000, 100, 100, 3)
print(y_train.shape,y_test.shape) # (14000,) (6000,)


#2. 모델 구성
model = Sequential()
model.add(Conv2D(40,(1,1), input_shape=(110,110,3),padding='same',strides=2))
model.add(Conv2D(45, (1,1), activation='relu',strides=1, padding='same'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(4,4))
model.add(Reshape(target_shape=(169,45)))
model.add(LSTM(units=5, input_shape=(169,45)))
model.add(Flatten())
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(20, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='sigmoid'))



#3. 컴파일 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) # accuracy, mse

start_time = time.time()
################
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
################
es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=10, verbose=1,
                   restore_best_weights=True)
################
import datetime

date = datetime.datetime.now()

date = date.strftime('%m%d_%H%M')

filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path_csv, 'k42_2_', date,'_', filename])

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode = 'auto',
    verbose= 1,
    save_best_only=True,
    filepath = filepath
)


model.fit(x_train, y_train, epochs=50,  batch_size=70,
                 verbose=1,
                 validation_split=0.3,
                 callbacks=[es,mcp]
                 )

end_time = time.time()

#4. 평가 예측
loss = model.evaluate(x_test, y_test)

y_submit = model.predict(xy_test[0][0])
y_pred = model.predict(x_test)
print(y_pred)

y_pred = np.round(y_pred)

accuracy_score = accuracy_score(y_test, y_pred)

submission_csv['label'] = (y_submit)

submission_csv.to_csv(path_csv + 'submission_0803_17.csv')

print('로스 :', loss)
print('acc_score :', accuracy_score)
print('소요시간 :', round(end_time - start_time), '초')

# LSTM
# 로스 : [0.6678836345672607, 0.5605999827384949]
# acc_score : 0.5606
# 소요시간 : 101 초