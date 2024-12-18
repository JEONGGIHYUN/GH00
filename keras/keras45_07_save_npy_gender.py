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

path_train = 'C:/TDS/ai5/_data/kaggle/Biggest gender/faces/'
# path_test = './_data/kaggle/dogs-vs-cats-redux-Kernels-edition/test/'
# path_csv = './_data/kaggle/dogs-vs-cats-redux-Kernels-edition/'

xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(120,120),
    batch_size=27167,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True,
)

np_path = 'C:/TDS/ai5/_data/_save_npy/'

np.save(np_path + 'keras45_07_x_train2.npy', arr=xy_train[0][0])
np.save(np_path + 'keras45_07_y_train2.npy', arr=xy_train[0][1])





