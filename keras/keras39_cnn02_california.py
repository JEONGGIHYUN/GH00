from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D,Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
#1. 데이터
dataset = fetch_california_housing()

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=3)

print(x_train.shape, x_test.shape) # (17544, 8) (3096, 8)

x_train = x_train.reshape(17544,2,2,2)
x_test = x_test.reshape(3096,2,2,2)

# x_train = x_train / 255.
# x_test = x_test / 255.

#2. 모델 구성
model = Sequential()

model.add(Conv2D(64,kernel_size=(1,1), input_shape=(2,2,2))) 
model.add(Conv2D(50, (1,1)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(25, activation='relu'))
model.add(Dense(15))
model.add(Dense(1))


#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')

start_time = time.time()
################
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
################
es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=10, verbose=1,
                   restore_best_weights=True)
################

################
model.fit(x_train, y_train, epochs=3000,  batch_size=32,
                 verbose=2,
                 validation_split=0.3,
                 callbacks=[es]
                 )
end_time = time.time()

#4. 평가 예측
loss = model.evaluate(x_test, y_test, verbose=0)
results = model.predict(x_test)
r2 = r2_score(y_test, results)
print('로스 :', loss)
print('r2스코어 :', r2)
print('소요시간 :', round(end_time - start_time), '초')

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

# -------------------------------
# 로스 : 0.6966645121574402
# r2스코어 : 0.4829765468585424

#[실습]
# R2 0.59 이상

# cpu 소요시간 : 14 초

# gpu 소요시간 : 20 초