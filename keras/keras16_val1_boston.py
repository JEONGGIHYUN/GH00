from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score


#1. 데이터
dataset = load_boston()
# print(dataset)
# print(dataset.DESCR) # 연습용으로만 사용하는 코드 이다 print(array.DESCR)
# print(dataset.feature_names) # 연습용 2 print(array.feature_names)


x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=1, shuffle=True)

'''
print(x)
print(x.shape) # (506, 13)
print(y)
print(y.shape) # (506, )
'''

#2. 모델 구성
model = Sequential()
model.add(Dense(150, input_dim=13))
model.add(Dense(30))
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(7))
model.add(Dense(1))


#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000,
          verbose=1,
          validation_split=0.3)

#4. 평가 예측
loss = model.evaluate(x_test, y_test, verbose=0)
results = model.predict(x_test)
print('로스 :', loss)
print('보스턴 집값 :', results)

r2 = r2_score(y_test, results)
print('r2스코어 :', r2)

# r2스코어 : 0.5899077941984204 state=10
# r2스코어 : 0.656526525424478 state=1754
# r2스코어 : 0.6982870443436436 state=350
# r2스코어 : 0.7301889651812639 state=350 0.7
# r2스코어 : 0.7358672801518396 state=350 0.8
# r2스코어 : 0.7612685477975188 state=1 train0.85 epochs3000
# r2스코어 : 0.7706839760687534 state=1 train0.85 epochs3000
# r2스코어 : 0.7486948843036088 state=1 traim0.85 epochs5000