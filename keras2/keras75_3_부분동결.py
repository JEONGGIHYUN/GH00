import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.applications import VGG16

vgg16 = VGG16(weights='imagenet', include_top=False,
              input_shape=(32,32,3)
              )

model = Sequential()

model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10,activation='softmax'))

#1. 전체 동결
# model.trainable = False

#2. 
# for layer in model.layers:
    # layer.trainable = False

#3. 부분동결
# print(model.layers)

# [<keras.engine.functional.Functional object at 0x0000022B61E65B50>,
#  <keras.layers.core.flatten.Flatten object at 0x0000022B61E583A0>,
#  <keras.layers.core.dense.Dense object at 0x0000022B61E8E5E0>,
#  <keras.layers.core.dense.Dense object at 0x0000022B61FF2A30>]

model.layers[0].trainable = False






model.summary()

print(len(model.weights))
print(len(model.trainable_weights))

import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns =['Layer Type', 'Layer Name', 'Layer Trainable'])
print(results)










