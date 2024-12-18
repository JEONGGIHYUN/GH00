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

# dataset['engine'] = data['engine'].str.split().str[0]
# dataset['engine_unit'] = data['engine'].str.split().str[1]



#1. 데이터
path1 = "C:/TDS/ai5/_data/jena/"
datasets = pd.read_csv(path1 + ".csv" )