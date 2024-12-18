# https://www.kaggle.com/datasets/maciejgronczynski/biggest-genderface-recognition-dataset/data

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,    
    vertical_flip=True,      
    width_shift_range=0.2,   
    height_shift_range=0.2,  
    rotation_range=4,        
    zoom_range=0.4,          
    shear_range=0.5,         
    fill_mode='nearest',     
)

# test_datagen = ImageDataGenerator(
#     rescale=1./255
# )

path_train = 'C:/TDS/ai5/_data/kaggle/Biggest gender/faces/woman/'
# path_train = 'C:/TDS/ai5/_data/kaggle/Biggest gender/faces/woman'

# train = train_datagen.flow_from_directory(
#     path_train,
#     target_size=(100,100),
#     batch_size=27167,
#     class_mode='binary',
#     color_mode='rgb',
#     shuffle=True,
# )
# train = train_datagen.flow_from_dataframe(
#     path_train,
#     target_size=(100,100),
#     batch_size=27167,
#     class_mode='binary',
#     color_mode='rgb',
#     shuffle=True,
# )


# man = train[0]
# woman = train[1]
# print(man.shape, woman.shape)



# np_path = 'C:/TDS/ai5/_data/kaggle/Biggest gender/save/'
np.save(np_path + 'man.npy', arr=man(label))

# np.save(np_path + 'woman.npy', arr=woman)



augment_size = 8189

randidx = np.random.randint(path_train, size=augment_size) # 60000, size=40000

x_augmented = path_train[randidx].copy()#  메모리 안전 

# y_augmented = y_train[randidx].copy()
# x_augmented = x_augmented.reshape()

x_augmented = train_datagen.flow(
    x_augmented,
    batch_size = augment_size,
    shuffle=False,
).next()[0]
 
# x_train = x_train.reshape(19997, 100, 100, 3)
# y_train = y_train.reshape(5000, 100, 100, 3)

x_train = np.concatenate((path_train, x_augmented), axis=0)

print(x_train.shape)

# y_train = np.concatenate((y_train, y_augmented), axis=0)


# #2. 모델 구성
# model = Sequential()
# model.add(Conv2D(60,(4,4), input_shape=(100,100,3)))
# model.add(Conv2D(60, (2,2), activation='relu', strides=2))
# model.add(Conv2D(60, (4,4), activation='relu'))
# model.add(Conv2D(60, (4,4), activation='relu'))
# model.add(Conv2D(60, (4,4), activation='relu'))
# model.add(MaxPooling2D(2,2))
# model.add(Flatten())
# model.add(Dense(120, activation='relu'))
# model.add(Dropout(0.4))
# model.add(Dense(60, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(1, activation='sigmoid'))

# #3. 컴파일 훈련
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) # accuracy, mse

# start_time = time.time()
# ################
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# ################
# ################
# es = EarlyStopping(monitor='val_loss', mode='min',
#                    patience=10, verbose=1,
#                    restore_best_weights=True)

# model.fit(x_train, y_train, epochs=30,  batch_size=100,
#                  verbose=1,
#                  validation_split=0.3,
#                  callbacks=[es]
#                  )

# end_time = time.time()

# #4. 평가 예측
# loss = model.evaluate(x_test, y_test)


# y_pred = model.predict(x_test)
# print(y_pred)

# y_pred = np.round(y_pred)

# accuracy_score = accuracy_score(y_test, y_pred)


# print('로스 :', loss)
# print('acc_score :', accuracy_score)
# print('소요시간 :', round(end_time - start_time), '초')




