import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


# 클래스와 인스턴스에 대해서 정리하여 깜지 한 장 만들어 제출

text = '나는 지금 현재 진짜 매우 매우 맛있는 김밥을 엄청 마구 마구 마구 먹었다.'

token = Tokenizer()
token.fit_on_texts([text])

print(token.word_index)
# {'마구': 1, '매우': 2, '나는': 3, '지금': 4, '현재': 5, '진짜': 6, '맛있는': 7, '김밥을': 8, '엄청': 9, '먹었다': 10, 'ㅁㅁㅁ': 11}
# 순서
# 1. 많이 나온 순서
# 2. 먼저 나온 순서 

print(token.word_counts)
# OrderedDict([('나는', 1), ('지금', 1), ('현재', 1), ('진짜', 1), ('매우', 2), ('맛있는', 1), ('김밥을', 1), ('엄청', 1), ('마구', 3), ('먹었다', 1), ('ㅁㅁㅁ', 1)])

x = token.texts_to_sequences([text])
print(x)
# [[3, 4, 5, 6, 2, 2, 7, 8, 9, 1, 1, 1, 10, 11]]

############ 원핫 3가지 만들기 ##################
# keras 원핫 인코딩
one_hot = to_categorical(x)
print(one_hot)
print(one_hot.shape)
# one_hot = one_hot
# print(one_hot.shape)
# x = one_hot[:,:,3:]
# x.fit(label.values.reshape(-1,1))

# x = x.reshape(1, 14, 9)
# print(x)
# print(x.shape)

# ohe = OneHotEncoder(sparse=False)
# print(ohe)

#첫번째는 많이 나오는 순서, 뒤에는 먼저 나오는 순서



#1 데이터 분석하고 trian data 하고 
# 랜덤 포레스트 모델 학습 = DNN 모델 썼다
# Regressor=회귀



# print(token.word_counts)
# OrderedDict([('성우하이텍', 1), ('떡상', 1), ('가즈아', 1), ('영차', 9)])

x = token.texts_to_sequences([text])

'''
x = pd.DataFrame(x)
print(x)
x = x.to_numpy()
print(x)
print(x.shape)
x = x.reshape(13,1)
print(x)

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
x = to_categorical(x)
print(x.shape)

# [[3, 4, 5, 2, 2, 6, 7, 8, 1, 1, 1, 1, 9]]
x = x[:, :, 1:]
print(x.shape)
x = x.reshape(14,9)

################# 원핫 3가지 맹글어봐!!############
#갯더미 원핫 인코더 

#x = pd.get_dummies(sum(x, []))




ohe = OneHotEncoder(sparse=False)
x = np.array(x).reshape(-1,1) 
ohe.fit(x)
x_encoded3 = ohe.transform(x)   
print(x_encoded3)


###################
x_encoded2 = ohe.fit_transform(x)
print(x_encoded2)

'''
###################
x = pd.get_dummies(pd.Series(np.array(x).reshape(-1,)))
print(x)


























































































