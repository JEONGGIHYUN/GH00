import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,    # 수평 뒤집기
    vertical_flip=True,      # 수직 뒤집기
    width_shift_range=0.1,   # 평행 이동
    height_shift_range=0.1,  # 평행 이동 수직
    rotation_range=5,        # 정해진 각도만큼 이미지 회전
    zoom_range=1.2,          # 축소 또는 확대
    shear_range=0.7,         # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환
    fill_mode='nearest'      # 데이터의 비어있는 곳을 가까운 데이터와 비슷한 값으로 채움 
)
test_datagen = ImageDataGenerator(
    rescale= 1./255
)

path_train = './_data/image/brain/train/'
path_test = './_data/image/brain/train/'

xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(200, 200),
    batch_size=10,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
)

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(200,200),
    batch_size=10,
    class_mode='binary',
    color_mode='grayscale',
)

print(xy_train)
# <keras.preprocessing.image.DirectoryIterator object at 0x000002637C597700>
print(xy_train.next()) # [0., 0., 0., 0., 1., 0., 1., 0., 1., 0.]
print(xy_train.next()) # [0., 1., 0., 0., 0., 1., 0., 1., 0., 1.]

print(xy_train[0])
print(xy_train[0][0])
print(xy_train[0][1])
# print(xy_train[0].shape) # AttributeError: 'tuple' object has no attribute 'shape'
print(xy_train[0][0].shape)
# print(xy_train[16]) # ValueError: Asked to retrieve element 16, but the Sequence has length 16
# print(xy_train[15][2]) # IndexError: tuple index out of range

print(type(xy_train)) # 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0])) # IndexError: tuple index out of range
print(type(xy_train[0][0])) # <class 'numpy.ndarray'>
print(type(xy_train[0][1])) # <class 'numpy.ndarray'>



























