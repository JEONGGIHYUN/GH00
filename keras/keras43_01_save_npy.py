# https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/overview

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D,Flatten, Dropout
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

start_time = time.time()

test_datagen = ImageDataGenerator(
    rescale= 1./255
)
# start_time = time.time()
path_train = './_data/kaggle/dogs-vs-cats-redux-kernels-edition/train/train/'
path_test = './_data/kaggle/dogs-vs-cats-redux-Kernels-edition/test/'
path_csv = './_data/kaggle/dogs-vs-cats-redux-Kernels-edition/'

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

np_path = 'C:/TDS/ai5/_data/_save_npy/'

np.save(np_path + 'keras43_01_x_train.npy', arr=xy_train[0][0])
np.save(np_path + 'keras43_01_y_train.npy', arr=xy_train[0][1])
np.save(np_path + 'keras43_01_x_test.npy', arr=xy_test[0][0])
np.save(np_path + 'keras43_01_y_test.npy', arr=xy_test[0][1])

end_time = time.time()

x_train, x_test, y_train, y_test = train_test_split(xy_train[0][0], xy_train[0][1], train_size=0.8, random_state=42)

# x_train = xy_train[0][0]
# y_train = xy_train[0][1]
# x_test = xy_test[0][0]
# y_test = xy_test[0][1]

# end_time = time.time()

print(x_train.shape,x_test.shape) # (14000, 100, 100, 3) (6000, 100, 100, 3)
print(y_train.shape,y_test.shape) # (14000,) (6000,)
print('소요시간 :', round(end_time - start_time), '초')
