from keras.preprocessing.text import Tokenizer
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Conv1D, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
from tensorflow.keras.utils import to_categorical

#1. 데이터
docs = [
    '너무 재미있다', '참 최고에요', '참 잘만든 영화예요',
    '추천하고 싶은 영화입니다.', '한 번 더 보고 싶어요.', '글쎄',
    '별로에요', '생각보다 지루해요', '연기가 어색해요',
    '재미없어요', '너무 재미없다.', '참 재밋네요.',
    '준영이 바보', '반장 잘생겼다', '태운이 또 구라친다',
]

x_pre = ['태운이 참 재미없다.']

y = labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0])

token = Tokenizer()
token.fit_on_texts(docs)
token1 = Tokenizer()
token1.fit_on_texts(x_pre)
print(token.word_index)
# {'참': 1, '너무': 2, '재미있다': 3, '최고에요': 4, '잘만든': 5, '영화예요': 6, '추천하고': 7, '싶은': 8, '영화입니다': 9, '한': 10, '번': 11, '더': 12, '보고': 13, '싶어요': 14, '글쎄': 15, '별로에요': 16, '생각보다': 17, '지루해요': 18, '연기가': 19, '어색해요': 20, '재미없어요': 21, '재미없다': 22, '재밋네요': 23, '준영이': 24, '바보': 25, ' 
# 반장': 26, '잘생겼다': 27, '태운이': 28, '또': 29, '구라친다': 30}

#  {'태': 1, '운': 2, '이': 3, '참': 4, '재': 5, '미': 6, '없': 7, '다': 8}


x = token.texts_to_sequences(docs)

y_pre = token.texts_to_sequences(x_pre)

print(x)
print(type(x)) # <class 'list'>
# [[2, 3], [1, 4], [1, 5, 6], [7, 8, 9], 
# [10, 11, 12, 13, 14], [15], 
# [16], [17, 18], [19, 20], 
# [21], [2, 22], [1, 23], 
# [24, 25], [26, 27], [28, 29, 30]]

print(y_pre)
print(type(y_pre)) # <class 'list'>
# [[1], [2], [3], [], [4], [], [5], [6], [7], [8], []]



# 변수안의 리스트의 최대 길이 찾기
'''
max_len = max(len(item) for item in x)
print('최대 길이 :',max_len)
'''

# 넘파이로 패딩
'''
for sentence in encoded:
    while len(sentence) < max_len:
        sentence.append(0)

padded_np = np.array(encoded)
padded_np
'''

from tensorflow.keras.preprocessing.sequence import pad_sequences #많이 사용할 것 같은 너낌
# from keras.utils import pad_sequences
x = pad_sequences(x,
                   # padding='pre','post'
                   maxlen=5,
                   #truncating='pre'
                   ) # pre: 앞으로 post:뒤로
# y = pad_sequences(y, maxlen=15)

y_pre = pad_sequences(y_pre,maxlen=31)

x_end = to_categorical(x)

y_end = to_categorical(y_pre)

# x_end = x_end.reshape()

print(x_end, x_end.shape) # (15, 5, 31)

print(y_end, y_end.shape) # (1, 31, 29)

# (xy == padded_np).all() # 결과가 같은지 확인하는 파이썬 코드

x_train, x_test, y_train, y_test = train_test_split(x_end, y, train_size=0.8, random_state=3)

print(x_train.shape,x_test.shape,y_train.shape,y_test.shape) # (12, 5, 31) (3, 5, 31) (12,) (3,)

# x_train = x_train.reshape(12, 5, 31)
exit()
y_end = y_end.reshape(1, 29, 31)
print(x_train.shape,y_train.shape)

#2. 모델 dnn

model = Sequential()
model.add(LSTM(units=6,input_shape=(5, 31)))
# model.add(Flatten())
model.add(Dense(128))
model.add(Dense(72, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=10, verbose=1,
                   restore_best_weights=True)

model.fit(x_train,y_train, epochs=100, batch_size=3000,
          verbose=2,
          validation_split=0.2,
          callbacks=[es])

#4.예측 평가

loss = model.evaluate(x_test, y_test)

print('로스 :', loss[0])
print('acc :', round(loss[1],3))


y_predict = model.predict(y_end)

y_predict = np.round(y_predict)
print(y_predict)
# print('acc_score :', accuracy_score)

