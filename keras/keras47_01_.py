# https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/overview

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D,Flatten, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import time
import os

path = 'C:/TDS/ai5/_data/image/mmmmmmmm/2.jpg'

img = load_img(path, target_size=(110,110))
print(img)

print(type(img))

arr = img_to_array(img)
print(arr)
print(arr.shape)
print(type(arr))

# 차원증가
img = np.expand_dims(arr, axis=0)
print(img.shape) # (1,100,100,3)

# start_time = time.time()
path_train = './_data/kaggle/dogs-vs-cats-redux-kernels-edition/train/train/'
path_test = './_data/kaggle/dogs-vs-cats-redux-Kernels-edition/test/'
path_csv = './_data/kaggle/dogs-vs-cats-redux-Kernels-edition/'

submission_csv = pd.read_csv(path_csv + 'sample_submission.csv', index_col=0)

np_path = 'C:/TDS/ai5/_data/_save_npy/'
# np.save(np_path + 'keras43_01_x_train.npy', arr=xy_train[0][0])
# np.save(np_path + 'keras43_01_y_train.npy', arr=xy_train[0][1])
# np.save(np_path + 'keras43_01_x_test.npy', arr=xy_test[0][0])
# np.save(np_path + 'keras43_01_y_test.npy', arr=xy_test[0][1])

x_train = np.load(np_path + 'keras43_01_x_train.npy')

y_train = np.load(np_path + 'keras43_01_y_train.npy')

x_d = np.load(np_path + 'keras43_01_x_test.npy')

# y_test = np.load(np_path + 'keras43_01_y_test.npy')


x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.8, random_state=42)

# x_train = xy_train[0][0]
# y_train = xy_train[0][1]
# x_test = xy_test[0][0]
# y_test = xy_test[0][1]

# end_time = time.time()

# print(x_train.shape,x_test.shape) # (14000, 100, 100, 3) (6000, 100, 100, 3)
# print(y_train.shape,y_test.shape) # (14000,) (6000,)


#2. 모델 구성

#3. 컴파일 훈련

#4. 평가 예측
print('================== 1. save.model 출력====================')
model = load_model('C:/TDS/ai5/_data/kaggle/dogs-vs-cats-redux-kernels-edition/좋흔데이터/k42_2_0804_0238_0009-0.6554.hdf5')

# loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)

y_submit = model.predict(img)
y_pred = model.predict(img)
print(y_pred)

y_pred = np.round(y_pred)

print(y_pred)

# [[0.5507339]]
# [[1.]]


