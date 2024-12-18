import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

tf.random.set_seed(333)
np.random.seed(333)

from tensorflow.keras.applications import VGG16

############### vgg16 디폴트 모델 ####################
# model = VGG16()
# model.summary()

#  input_1 (InputLayer)        [(None, 224, 224, 3)]     0

#  predictions (Dense)         (None, 1000)              4097000

# =================================================================
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0
# _________________________________________________________________

######################################################

# model = VGG16(weights='imagenet',
#               include_top=True,
#               input_shape=(224, 224, 3)
#               )

# model.summary()

######################################################
# include를 False로 해 주면 fc layer을 날려준다.
# model = VGG16(weights='imagenet',
#               include_top=False,
#               input_shape=(224, 224, 3)
#               )

# model.summary()

######################################################
# include를 False로 해 주면 fc layer을 날려준다.
# include를 False로 해주면 input_shape를 하고싶은 shape로 지정해 줄 수 있다.
# model = VGG16(weights='imagenet',
            #   include_top=False,
            #   input_shape=(100, 100, 3)
            #   )

# model.summary()

######################################################