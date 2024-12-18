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

start_time = time.time()
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

x_test = np.load(np_path + 'keras43_01_x_test.npy')

y_test = np.load(np_path + 'keras43_01_y_test.npy')

# print(x_train)
print(x_train.shape) # (25000, 110, 110, 3)
# print(y_train)
print(y_train.shape) # (25000,)
# print(x_test)
print(x_test.shape) # (12500, 110, 110, 3)
# print(y_test)
print(y_test.shape) # (12500,)

end_time = time.time()

# x_train, x_test, y_train, y_test = train_test_split(xy_train[0][0], xy_train[0][1], train_size=0.8, random_state=42)

# x_train = xy_train[0][0]
# y_train = xy_train[0][1]
# x_test = xy_test[0][0]
# y_test = xy_test[0][1]

# end_time = time.time()

# print(x_train.shape,x_test.shape) # (14000, 100, 100, 3) (6000, 100, 100, 3)
# print(y_train.shape,y_test.shape) # (14000,) (6000,)
print('소요시간 :', round(end_time - start_time), '초')