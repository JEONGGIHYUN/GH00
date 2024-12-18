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

path1 = "C:/TDS/ai5/_data/jena/"
# datasets = pd.read_csv(path1 + "jena_climate_2009_2016.csv" )
data1 = pd.read_csv(path1 + "월일1.csv")
data2 = pd.read_csv(path1 + '시간1.csv')
data3 = pd.read_csv(path1 + "분초1.csv")
data4 = pd.read_csv(path1 + "년도뺀거.csv")



daa = pd.concat([data1,data2,data3,data4],axis=1,ignore_index=True)
daa.columns = ['D','M','H','M','S','p (mbar)','T (degC)','Tpot (K)','Tdew (degC)','rh (%)','VPmax (mbar)','VPact (mbar)','VPdef (mbar)','sh (g/kg)','H2OC (mmol/mol)','rho (g/m**3)','wv (m/s)','max. wv (m/s)','wd (deg)']

print(daa)
daa.to_csv(path1 + 'cs2.csv')






















