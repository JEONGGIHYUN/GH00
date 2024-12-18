import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D,Flatten, Reshape, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time

train_datagen = ImageDataGenerator(
    rescale=1./255
)
test_datagen = ImageDataGenerator(
    rescale= 1./255
)

path_train = 'C:/TDS/ai5/_data/image/horse_human/'
path_test = 'C:/TDS/ai5/_data/image/horse_human/'

xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(100, 100),
    batch_size=20000,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True,
)

# xy_test = test_datagen.flow_from_directory(
#     path_test,
#     target_size=(100, 100),
#     batch_size=20000,
#     class_mode='binary',
#     color_mode='rgb',
# )

np_path = 'C:/TDS/ai5/_data/_save_npy/'

np.save(np_path + 'keras45_02_x_train.npy', arr=xy_train[0][0])
np.save(np_path + 'keras45_02_y_train.npy', arr=xy_train[0][1])

x_train, x_test, y_train, y_test = train_test_split(xy_train[0][0], xy_train[0][1], train_size=0.7, random_state=3)





#2. 모델 구성
model = Sequential()
model.add(Conv2D(5,(4,4), input_shape=(100,100,3)))
model.add(Conv2D(5, (4,4), activation='relu'))
model.add(MaxPooling2D(4,4))
model.add(Reshape(target_shape=(529,5)))
model.add(LSTM(units=5, input_shape=(529,5)))
model.add(Flatten())
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#3. 컴파일 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) # accuracy, mse

start_time = time.time()
################
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
################
es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=3, verbose=1,
                   restore_best_weights=True)
################

model.fit(x_train, y_train, epochs=30,  batch_size=100,
                 verbose=1,
                 validation_split=0.3,
                 callbacks=[es]
                 )

end_time = time.time()

#4. 평가 예측
loss = model.evaluate(x_test, y_test)


y_pred = model.predict(x_test)
print(y_pred)

y_pred = np.round(y_pred)

accuracy_score = accuracy_score(y_test, y_pred)


print('로스 :', loss)
print('acc_score :', accuracy_score)
print('소요시간 :', round(end_time - start_time), '초')

# LSTM
# 로스 : [0.687152087688446, 0.5307443141937256]
# acc_score : 0.5307443365695793
# 소요시간 : 6 초