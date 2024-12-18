from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score



#1. 데이터
datasets = load_diabetes()

x = datasets.data
y = datasets.target

# print(x.shape, y.shape) (442, 10) (442, )

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=7251)

#2. 모델 구성
model = Sequential()
model.add(Dense(251, input_dim=10))
model.add(Dense(141))
model.add(Dense(171))
model.add(Dense(14))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000,
          verbose=0,
          validation_split=0.3)

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
r2 = r2_score(y_test, results)
print('로스 :', loss)
print('r2스코어 :', r2)

# 로스 : 2607.511962890625
# r2스코어 : 0.5607322201160404 train .9 random 5 epochs 1000

# 로스 : 2431.83935546875
# r2스코어 : 0.5818174376445646 train .9 random 9 epochs 1000

# 로스 : 2407.46728515625
# r2스코어 : 0.5860084929864275 train .9 random 9 epochs 1000

# 로스 : 2405.62158203125
# r2스코어 : 0.5863258357446803  train .9 random 9 epochs 1000

# 로스 : 2832.931884765625
# r2스코어 : 0.6184719992107199 train .9 random 52151 epochs 1000

# 로스 : 2817.88916015625
# r2스코어 : 0.6204978866983641 train .9 random 52151 epochs 1000

# 로스 : 2803.8544921875
# r2스코어 : 0.6223879836278867 train .9 random 52151 epochs 1000

# 로스 : 2120.6298828125 
# r2스코어 : 0.6278116967478458 train .9 random 7251 epochs 1000

# 로스 : 2056.626708984375
# r2스코어 : 0.6390448036614433 train .9 random 7251 epochs 1000

# 로스 : 2039.3199462890625
# r2스코어 : 0.6420822125482439 train .9 random 7251 epochs 1000







