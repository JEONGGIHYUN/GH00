from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img # 이미지 불러오기
from tensorflow.keras.preprocessing.image import img_to_array # 불러온 이미지 수치화
import numpy as np
import matplotlib.pyplot as plt

path = 'C:/TDS/ai5/_data/image/mmmmmmmm/1.jpg'

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

################ 증폭 #################
datagen = ImageDataGenerator(
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

it = datagen.flow(img,
             batch_size=1,
             )
# print(it) # <keras.preprocessing.image.NumpyArrayIterator object at 0x000001559EA65BB0>

print(it.next())

fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(5, 5))

for i in range(5):
    batch = it.next()
    print(batch.shape) # (1, 100, 100, 3)
    batch = batch.reshape(100,100,3)
    
    ax[i].imshow(batch)
    ax[i].axis('off')

plt.show()





























