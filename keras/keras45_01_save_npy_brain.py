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

train_datagen = ImageDataGenerator(
    rescale=1./255,
)
test_datagen = ImageDataGenerator(
    rescale= 1./255
)

path_train = './_data/image/brain/train/'
path_test = './_data/image/brain/train/'

xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(200, 200),
    batch_size=160,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
)

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(200,200),
    batch_size=160,
    class_mode='binary',
    color_mode='grayscale',
)

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

print(x_train.shape,x_test.shape) # (160 200 200 1)(10,)
print(y_train.shape,y_test.shape) # 

np_path = 'C:/TDS/ai5/_data/_save_npy/'
np.save(np_path + 'keras45_01_x_train.npy', arr=xy_train[0][0])
np.save(np_path + 'keras45_01_y_train.npy', arr=xy_train[0][1])
np.save(np_path + 'keras45_01_x_test.npy', arr=xy_test[0][0])
np.save(np_path + 'keras45_01_y_test.npy', arr=xy_test[0][1])


# #2. 모델 구성
# model = Sequential()
# model.add(Conv2D(30,(2,2), input_shape=(200,200,1),padding='same'))
# model.add(Conv2D(20, (2,2), activation='relu',padding='same'))
# model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dense(30, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# #3. 컴파일 훈련
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) # accuracy, mse

# ################
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# ################
# es = EarlyStopping(monitor='val_loss', mode='min',
#                    patience=10, verbose=1,
#                    restore_best_weights=True)
# ################

# model.fit(x_train, y_train, epochs=30,  batch_size=1000,
#                  verbose=2,
#                  validation_split=0.3,
#                  callbacks=[es]
#                  )

# #4. 평가 예측
# loss = model.evaluate(x_test, y_test)


# y_pred = model.predict(x_test)
# print(y_pred)

# y_pred = np.round(y_pred)

# accuracy_score = accuracy_score(y_test, y_pred)


# print('로스 :', loss)
# print('acc_score :', accuracy_score)

