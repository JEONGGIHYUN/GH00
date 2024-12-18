import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.applications import VGG16

model = VGG16(weights='imagenet', include_top=True,
              input_shape=(224,224,3)
              )

model.layers[20].trainable = False


# =================================================================
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0
# _________________________________________________________________
model.summary()


print(len(model.weights))
print(len(model.trainable_weights))

import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns =['Layer Type', 'Layer Name', 'Layer Trainable'])
print(results)

