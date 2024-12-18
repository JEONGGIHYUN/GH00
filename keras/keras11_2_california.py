from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. 데이터
dataset = fetch_california_housing()

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=32343, shuffle=True)

print(x)
print(y)
print(x.shape, y.shape) #(20640, 8) (20640, )

#2. 모델 구성
model = Sequential()
model.add(Dense(341, input_dim=8))
model.add(Dense(125))
model.add(Dense(12))
model.add(Dense(25))
model.add(Dense(15))
model.add(Dense(1))


#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200)


#4. 평가 예측
loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
r2 = r2_score(y_test, results)
print('로스 :', loss)
print('r2스코어 :', r2)

# 로스 : 0.5936856269836426
# r2스코어 : 0.5581232931681027

# 로스 : 0.5947655439376831
# r2스코어 : 0.5585999172838747 train 0.9 random 3 epochs 200




#[실습]
# R2 0.59 이상
