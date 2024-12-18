from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. 데이터
dataset = fetch_california_housing()

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=3, shuffle=True)
################################################
from sklearn.preprocessing import MinMaxScaler , StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
scaler = RobustScaler()

scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)
################################################
# print(x)
# print(y)
# print(x.shape, y.shape) #(20640, 8) (20640, )

################
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
################



#4. 평가 예측

print('================== 1. save.model 출력====================')
model = load_model('C:/TDS/ai5/study/_save/keras30_mcp/k30_02_0726_1939_0005-0.9006.hdf5')

loss = model.evaluate(x_test, y_test, verbose=0)
results = model.predict(x_test)
r2 = r2_score(y_test, results)
print('로스 :', loss)
print('r2스코어 :', r2)

# 로스 : 0.5936856269836426
# r2스코어 : 0.5581232931681027

# 로스 : 0.5947655439376831
# r2스코어 : 0.5585999172838747 train 0.9 random 3 epochs 200

# 로스 : 0.5959801077842712 
# r2스코어 : 0.557698499581295 train 0.9 random 3 epochs 200

# 로스 : 0.5961485505104065
# r2스코어 : 0.5575735625525535

# MaxAbsScaler
# 로스 : 0.6150590181350708
# r2스코어 : 0.5435392913050829

# 로스 : 0.5986493825912476
# r2스코어 : 0.5557174838885757

# 로스 : 0.6089686155319214
# r2스코어 : 0.5480590906630616

# RobustScaler
# 로스 : 0.6855686902999878
# r2스코어 : 0.4912110596771896

# 로스 : 0.6074805855751038
# r2스코어 : 0.549163591826519

# 로스 : 0.601770281791687
# r2스코어 : 0.5534014535510928

#[실습]
# R2 0.59 이상