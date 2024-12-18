#2048개의 컬럼에 14개의 컬럼을 어떻게 합치냐가 이 게임의 승부가 된다.

import pandas as pd
import numpy as np
import os
import random

from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Conv1D, Bidirectional, Dense, Embedding, Flatten
from tensorflow.keras.callbacks import EarlyStopping

path = 'C:/ai5/_data/dacon/신약개발/'

CFG = {
    'NBITS':2048,
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
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=CFG['NBITS'])
        return np.array(fp)
    else:
        return np.zeros((CFG['NBITS'],))
    
    # 학습 ChEMBL 데이터 로드
chembl_data = pd.read_csv(path + 'train.csv')  # 예시 파일 이름
chembl_data.head()

train = chembl_data[['Smiles', 'pIC50']]
train['Fingerprint'] = train['Smiles'].apply(smiles_to_fingerprint)

train_x = np.stack(train['Fingerprint'].values)
train_y = train['pIC50'].values

# 학습 및 검증 데이터 분리
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

print(train_x.shape)
print(train_y.shape)
print(val_x.shape)
print(val_y.shape)


# 랜덤 포레스트 모델 학습
# model = RandomForestRegressor(random_state=CFG['SEED'])
model = Sequential()
model.add(Embedding(2048, 10)) 
model.add(Conv1D(10, 2))
model.add(Flatten())
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))

model.summary()

def pIC50_to_IC50(pic50_values):
    """Convert pIC50 values to IC50 (nM)."""
    return 10 ** (9 - pic50_values)

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics = ['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode = 'min',
    patience = 10,
    verbose=1,
    restore_best_weights=True
)


model.fit(train_x, train_y, epochs=1, batch_size=16, callbacks=[es])

#4. 평가 예측
loss = model.evaluate(val_x, val_y)



# Validation 데이터로부터의 학습 모델 평가
val_y_pred = model.predict(val_x)
mse = mean_squared_error(pIC50_to_IC50(val_y), pIC50_to_IC50(val_y_pred))
print(val_y_pred.shape)

print(pIC50_to_IC50(val_y).shape, pIC50_to_IC50(val_y_pred).shape)


rmse = np.sqrt(mse)


print(f'RMSE: {rmse}')

# test = pd.read_csv(path + 'test.csv')
# test['Fingerprint'] = test['Smiles'].apply(smiles_to_fingerprint)

# test_x = np.stack(test['Fingerprint'].values)

# test_y_pred = model.predict(test_x)

# submit = pd.read_csv(path + 'sample_submission.csv')
# submit['IC50_nM'] = pIC50_to_IC50(test_y_pred)
# submit.head()

# submit.to_csv(path + 'baseline_submit.csv', index=False)