# https://www.kaggle.com/c/otto-group-product-classification-challenge/leaderboard?tab=public
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder
import time

path = 'C:/TDS/ai5/_data/kaggle/otto-group-product-classification-challenge/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)

test_csv = pd.read_csv(path + 'test.csv', index_col=0)

submission_csv = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)

# train_csv.info()
# test_csv.info()
# print(train_csv['target'].value_counts())
# train_csv['target'] = train_csv['target'].replace({'Class_1' : 1, 'Class_1' : 1, 'Class_2' : 2, 'Class_3' : 3, 'Class_4' : 4, 'Class_5' : 5, 'Class_6' : 6, 'Class_7' : 7, 'Class_8' : 8, 'Class_9' : 9, })


############################################
x = train_csv.drop(['target'], axis=1)
# print(x.shape) # (61878, 93)

y = train_csv['target']
# print(y.shape) # (61878,)
###################################
y = pd.get_dummies(y)
# print(y)
# print(y.shape)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=3, shuffle=True, stratify=y)

###############################################
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
scaler = RobustScaler()

# scaler.fit(x_train)
# scaler.transform(x_train)
x_train = scaler.fit_transform(x_train)
scaler.transform(x_test)

###############################################
# print(x_train)
# print(np.min(x_train), np.max(x_train)) #
# print(np.min(x_test), np.max(x_test)) # 

#2. 모델 구성

#3. 컴파일 훈련

#4. 예측 평가

print('================== 1. save.model 출력====================')
model = load_model('C:/TDS/ai5/study/_save/keras30_mcp/k30_13_0726_1950_0021-0.5901.hdf5')

loss = model.evaluate(x_test, y_test)
y_predict = np.around(model.predict(x_test))
y_submit = model.predict(test_csv)
accuracy_score = accuracy_score(y_test, y_predict)

submission_csv[['Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9']] = y_submit
r2 = r2_score(y_test, y_predict)

print('r2_score : ', r2)
print('loss :', loss)
print('acc score :', accuracy_score)

# submission_csv.to_csv(path + 'submission_0725_16_56.csv')

# r2_score :  0.47041254628230056
# loss : [0.6029521226882935, 0.7807046175003052]
# acc score : 0.748868778280543

# r2_score :  0.4792573162461485
# loss : [0.5797273516654968, 0.7857950925827026]
# acc score : 0.750242404654169

# r2_score :  0.4616540108040687
# loss : [1.0574525594711304, 0.7874919176101685]
# acc score : 0.7790885585003232

# r2_score :  0.444643147690851
# loss : [0.6943469643592834, 0.7826438546180725]
# acc score : 0.7625242404654169
# time : 35.9 초

# r2_score :  0.48940098576306745
# loss : [0.6691646575927734, 0.791047215461731]
# acc score : 0.7752908855850033
# time : 37.04 초

# r2_score :  0.4449872089022284
# loss : [0.611876368522644, 0.7728399038314819]
# acc score : 0.7167097608274079
# time : 2.59 초

# r2_score :  0.4611566087009949
# loss : [0.6050901412963867, 0.7775802612304688]
# acc score : 0.7251131221719457
# time : 2.56 초

# MaxAbsScaler
# r2_score :  0.43892688161180626
# loss : [0.6179496049880981, 0.773270845413208]
# acc score : 0.708090928679164
# time : 2.56 초

# r2_score :  0.4346305124885742
# loss : [0.6224663853645325, 0.7712239027023315]
# acc score : 0.717948717948718
# time : 2.61 초

# RobustScaler
# r2_score :  0.436824595198675
# loss : [0.6232938766479492, 0.7684766054153442]
# acc score : 0.720318896789485
# time : 2.56 초

# r2_score :  0.43309573209832386
# loss : [0.6187971234321594, 0.773216962814331]
# acc score : 0.7066903684550744
# time : 2.56 초