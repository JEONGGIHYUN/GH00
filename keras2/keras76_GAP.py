# Global Average Pooling = 전체를 평균값으로 잡아서 풀링한다.
# Flatten의 대용이다
# 피쳐의 갯수만큼 진행한다.

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
import tensorflow as tf
import time
from sklearn.metrics import accuracy_score, r2_score

tf.random.set_seed(333)
np.random.seed(333)

from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train / 255.
x_test = x_test / 255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



vgg16 = VGG16(weights='imagenet',
              include_top=False,
              input_shape=(224, 224, 3)
              )
# vgg16.trainable = False # 동결건조

model = Sequential()
model.add(vgg16)
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

model.summary()


# r2_score :  -0.046333392451417134
# loss : [1.6453238725662231, 0.4269999861717224]
# acc score : 0.1115