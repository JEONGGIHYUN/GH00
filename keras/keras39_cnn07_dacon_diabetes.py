# https://dacon.io/competitions/official/236068/overview/description

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input,Conv2D,Flatten
from tensorflow.keras.callbacks import EarlyStopping
import time


#1. 데이터
path = './_data/dacon/diabetes/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
# print(train_csv) # [652 rows x 9 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=0)
# print(test_csv) # [116 rows x 8 columns]

submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)
# print(submission_csv)

# print(train_csv.shape) # (652, 9)
# print(test_csv.shape) # (116, 8)
# print(submission_csv.shape) # (116, 1)

# print(train_csv.columns)
# Index(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
    #    'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'],
    #   dtype='object')

x = train_csv.drop(['Outcome'], axis=1)

y = train_csv['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=3)

print(x_train.shape, x_test.shape) # (554, 8) (98, 8)

x_train = x_train.to_numpy()
x_test = x_test.to_numpy()

x_train = x_train.reshape(554,2,2,2)
x_test = x_test.reshape(98,2,2,2)

# x_train = x_train / 255.
# x_test = x_test / 255.

#2. 모델 구성
model = Sequential()

model.add(Conv2D(64,kernel_size=(2,2), input_shape=(2,2,2))) 
model.add(Conv2D(50, (1,1)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#3. 컴파일 훈련

model.compile(loss='mse', optimizer='adam', metrics='acc') # accuracy, mse

start_time = time.time()

################
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
################
es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=10, verbose=1,
                   restore_best_weights=True)
################

model.fit(x_train, y_train, epochs=3000,  batch_size=32,
                 verbose=2,
                 validation_split=0.3,
                 callbacks=[es]
                 )

end_time = time.time()

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
y_pred = np.round(model.predict(x_test))
acc = accuracy_score(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('로스 :', loss)
print('소요시간 :', round(end_time - start_time), '초')



# 로스 : [0.2036416232585907, 0.6989796161651611]

# 로스 : [0.20923519134521484, 0.6887755393981934]

# 로스 : [0.21842415630817413, 0.6938775777816772]

# MaxAbsScaler
# 로스 : [0.30629029870033264, 0.6224489808082581]

# 로스 : [0.2141127735376358, 0.6836734414100647]

# RobustScaler
# 로스 : [0.3571428656578064, 0.6428571343421936]

# 로스 : [0.21126089990139008, 0.6938775777816772]
# -----------------------------------------------------
# 로스 : [0.3430812358856201, 0.6122449040412903]

# cpu 소요시간 : 1 초

# gpu 소요시간 : 3 초