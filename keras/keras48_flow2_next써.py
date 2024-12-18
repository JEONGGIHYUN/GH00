from tensorflow.keras.datasets import fashion_mnist
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

################ 증폭 #################
train_datagen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip=True,    # 수평 뒤집기
    vertical_flip=True,      # 수직 뒤집기
    width_shift_range=0.2,   # 평행 이동
    # height_shift_range=0.2,  # 평행 이동 수직
    rotation_range=15,        # 정해진 각도만큼 이미지 회전
    # zoom_range=0.4,          # 축소 또는 확대
    # shear_range=0.7,         # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환
    fill_mode='nearest',     # 데이터의 비어있는 곳을 가까운 데이터와 비슷한 값으로 채움 
)

augment_size = 100

# print(x_train[0].shape) # (28, 28)

# aaa = np.tile(x_train[0], augment_size).reshape(-1, 28, 28, 1),

# print(aaa[0].shape) # (100, 28, 28, 1)

xy_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1, 28, 28, 1),
    np.zeros(augment_size),
    batch_size = augment_size,
    shuffle=False
).next()

print(xy_data)
print(type(xy_data)) # <class 'tuple'>
# print(xy_data.shape) # AttributeError: 'tuple' object has no attribute 'shape'
print(len(xy_data)) # 2
# plt.imshow(x_train[0], cmap='gray')
# plt.show()
print(xy_data[0].shape) # (100, 28, 28, 1)
print(xy_data[1].shape) # (100, )

plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7, 7, i+1)
    plt.imshow(xy_data[0][i], cmap='gray')
    
plt.show()























