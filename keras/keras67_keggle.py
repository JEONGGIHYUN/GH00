import pandas as pd
import numpy as np
import os
import random
import datetime

from sklearn.decomposition import PCA
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv1D,LSTM,SimpleRNN,GRU,Flatten,MaxPooling1D,Embedding,Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import RobustScaler, StandardScaler, Normalizer
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

CFG = {
    'NBITS':1024,
    'SEED':42,
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(CFG['SEED']) # Seed 고정

# SMILES 데이터를 분자 지문으로 변환
def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        # fp = AllChem.GetMorganFingerprintAsBitVect(mol, 1, nBits=CFG['NBITS'])
        fp = GetMorganGenerator(radius=2, fpSize=CFG['NBITS']).GetFingerprint(mol)
        return np.array(fp)
    else:
        return np.zeros((CFG['NBITS'],))

# 학습 ChEMBL 데이터 로드
path = 'C:/ai5/_data/dacon/신약개발/'
chembl_data = pd.read_csv(path + 'train.csv')  # 예시 파일 이름
print(chembl_data.head(10))
print(chembl_data.shape) # (1952, 15)

train = chembl_data[['Smiles','pIC50']]
train['Fingerprint'] = train['Smiles'].apply(smiles_to_fingerprint)



train_x = np.stack(train['Fingerprint'].values)
train_y = train['pIC50'].values



print(train_x.shape) # (1952, 144)
print(train_y.shape) # (1952,)
# print(train_x)

# pading
# train_x = pad_sequences(train_x,
#                    padding='pre',
#                    maxlen=20,
#                    #truncating='pre'
#                    )

# 학습 및 검증 데이터 분리
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=4567)

scaler = Normalizer()
train_x = scaler.fit_transform(train_x)
val_x = scaler.transform(val_x)

pca = PCA(n_components=512)
train_x = pca.fit_transform(train_x)
val_x = pca.transform(val_x)

print(train_x,val_x.shape) # (1366, 144) (586, 144) #pad했을때 (1366, 20) (586, 20)
print(train_y.shape,val_y.shape) # (1366,) (586,) #pad했을때 (1366,) (586,)

# import matplotlib.pyplot as plt
# plt.plot(train_x)
# plt.grid()
# plt.show()

# exit()


# # 랜덤 포레스트 모델 학습
# model = RandomForestRegressor(random_state=CFG['SEED'])
# model.fit(train_x, train_y)

# def pIC50_to_IC50(pic50_values):
#     """Convert pIC50 values to IC50 (nM)."""
#     return 10 ** (9 - pic50_values)


#2. 모델
model = Sequential()
model.add(Conv1D(filters=512,kernel_size=2,input_shape=(512,1),activation='relu', padding='same'))
model.add(Conv1D(filters=512,kernel_size=2,activation='relu'))
# model.add(Flatten())
# model.add(LSTM(units=30,input_shape=(512,1),activation='relu'))
model.add(Flatten())
# model.add(Dense(300, activation='relu'))
# model.add(Dropout(0,2))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(50, activation='relu'))
model.add(Dense(1))

#2-1. 함수형 모델

def pIC50_to_IC50(pic50_values):
    """Convert pIC50 values to IC50 (nM)."""
    return 10 ** (9 - pic50_values)

model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=150, verbose=1,
                   restore_best_weights=True)

date = datetime.datetime.now()

date = date.strftime("%m%d.%H%M")

path1 = 'C:/TDS/ai5/_save/신약/'
filename = '{epoch:04d}_{val_loss:.4f}.hdf5'
filepath = "".join([path1,'신약', date, '__epo__', filename])

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

model.fit(train_x, train_y, 
          epochs=300,
          verbose=1,
          callbacks=[es,mcp],
          validation_split=0.2
          )

# Validation 데이터로부터의 학습 모델 평가
val_y_pred = model.predict(val_x)
mse = mean_squared_error(pIC50_to_IC50(val_y), pIC50_to_IC50(val_y_pred))
# mse = mean_squared_error((val_y), (val_y_pred))
rmse = np.sqrt(mse)

print('loss :', mse)
print(f'RMSE: {rmse}')

test = pd.read_csv(path + 'test.csv')
test['Fingerprint'] = test['Smiles'].apply(smiles_to_fingerprint)

test_x = np.stack(test['Fingerprint'].values)

test_x = scaler.transform(test_x)
test_x = pca.transform(test_x)

test_y_pred = model.predict(test_x)

submit = pd.read_csv(path + 'sample_submission.csv')
submit['IC50_nM'] = pIC50_to_IC50(test_y_pred)
submit.head()

submit.to_csv(path + 'baseline19_submit.csv', index=False,)#float_format='%.14f')

# 랜덤 포레스트 학습모델
# 2274.669401784073
# 

# 모델 시퀀셜 학습 모델
# 812232747.8715354
# 1473189400.6100678 pad
# RMSE: 7.448668465268754
# RMSE: 2421.150891869797
# RMSE: 2391.224094853463
# RMSE: 1158.3271015155876 LSTM
# RMSE: 1362.7468406171809 GRU
# RMSE: 1128.0804856002908 GRU