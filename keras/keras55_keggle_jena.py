# https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016

import numpy as np

# y는 T (degC) 로 잡아라.

# 자르는거는 맘대로 ?

# 2016.12.31 00:10:00
# 2017.01.01 00:00:00 
#144개 맞추기 

import pandas as pd
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
import time
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터 분리
path1 = "C:/TDS/ai5/_data/jena/"
# datasets = pd.read_csv(path1 + "jena_climate_2009_2016.csv")#,index_col=0)
datasets = pd.read_csv(path1 + "cs2.csv" ,index_col=0)

# datasets.to_csv(path1 + '년도뺀거.csv', index=False)

#2. 데이터 분리 월일
# data = datasets['Date Time'].str.split('.',expand=True)
# data.to_csv(path1 + '일월.csv')
# data1 = pd.read_csv(path1 + '일월.csv', index_col=[0,3])
# data1.to_csv(path1 + '일월1.csv',index=False)

#3. 데이터 분리 분초
# data = datasets['Date Time'].str.split(':',expand=True)
# data.to_csv(path1 + '분초.csv')
# data1 = pd.read_csv(path1 + '분초.csv', index_col=[0,1])
# data1.to_csv(path1 + '분초1.csv',index=False)

# #4. 데이터 분리 시간 (귗낳 안해 대려버둟마ㅓㅇㄹ마ㅓㄷ)
# data = datasets['Date Time'].str.split(',',expand=True)
# data.to_csv(path1 + '시간.csv')
# data1 = pd.read_csv(path1 + '시간.csv', index_col=0)
# data1.to_csv(path1 + '시간1.csv',index=False)

# data1.to_csv(path1 + '시간1.csv',index=False)

# print(datasets.shape)   
# print(datasets.columns)
# print(datasets)


# data.to_csv(path1 + 'ddd.csv',index=False)
# data = pd.read_csv(path1 + 'ddd.csv')
# data = data['0'].str.split('.'' ',expand=True)

# print(data.shape)   
# print(data.columns)
# print(data)

# data.to_csv(path1 + 'sss.csv',index=False)

#------------------------------------------------------------------------------------------

print(datasets.shape) #(420551, 19)
a = datasets[:-144]
print(a.shape)      # (420407, 19)
print(a)
y_cor = datasets[-144:]['T (degC)']            # 예측치 정답
print(y_cor.shape)    # (144,)

x_predict = datasets[-144:]
print(x_predict.shape)      # (144, 19)

size = 864                    # 144 * 6 144 * 19

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size)
print(bbb)
print(bbb.shape)        # (419544, 864, 14)

x = bbb[:, : -144, ]
y = bbb[:, -144 : , 1]       # T 데이터 

print(x.shape, y.shape) # (419544, 720, 14) (419544, 144)
print(y)

# 48093 52560











































































































































