import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.utils import to_categorical

text1 = '나는 지금 진짜 진짜 매우 매우 맛있는 김밥을 엄청 마구 마구 마구 마구 먹었다.'
text2 = '태운이는 선생을 괴ㅗ빈다. 준영이는 못생겼다. 사영이는 마구 마구 더 못생겼다.'

#만들기
token = Tokenizer()
token.fit_on_texts([text1, text2])

print(token.word_index)

print(token.word_counts)

x = token.texts_to_sequences([text1])
xx = token.texts_to_sequences([text2])

x = pd.DataFrame(x)
xx = pd.DataFrame(xx)
xy = pd.concat([x,xx], axis=1)

one_hot = to_categorical(xy)
print(one_hot)
# print(xy)
# xy = xy.to_numpy()
# print(xy)
# print(xy.shape)
# xy = xy.reshape(14, 2)
# print(x)