import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import time

path = 'C:/TDS/ai5/_data/image/mmmmmmmm/2.jpg'

img = load_img(path, target_size=(100,100))
print(img)

print(type(img))

arr = img_to_array(img)
print(arr)
print(arr.shape)
print(type(arr))

# 차원증가
img = np.expand_dims(arr, axis=0)
print(img.shape) # (1,100,100,3)


np_path = 'C:/TDS/ai5/_data/_save_npy/'

x_train = np.load(np_path + 'keras45_07_x_train_gender.npy')

y_train = np.load(np_path + 'keras45_07_y_train_gender.npy')

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.8, random_state=4)

#2. 모델 구성

#3. 컴파일 훈련

#4. 평가 예측
print('================== 1. save.model 출력====================')
model = load_model('C:/TDS/ai5/_data/_save_npy/k45_가중치_0805_1540_0010-0.6059.hdf5')

y_pred = model.predict(img)

y_pred = np.round(y_pred,)
print(y_pred)

# [[1.]]