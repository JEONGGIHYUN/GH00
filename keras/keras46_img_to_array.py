from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img # 이미지 불러오기
from tensorflow.keras.preprocessing.image import img_to_array # 불러온 이미지 수치화
import numpy as np

<<<<<<< HEAD
path = 'C:/TDS/ai5/_data/image/mmmmmmmm/1.jpg'
=======
path = 'C:/ai5/_data/image/mmmmmmmm/1.jpg'
>>>>>>> b0025fd (Commit on 2024-07-13)

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