from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
import matplotlib.pyplot as plt


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
start_time = time.time()

model.compile(loss='mse', optimizer='adam')

hist = model.fit(x_train, y_train, epochs=1000,
          verbose=0,
          validation_split=0.3)

end_time = time.time()

#4. 평가 예측
loss = model.evaluate(x_test, y_test)

results = model.predict(x_test)

r2 = r2_score(y_test, results)

print('로스 :', loss)
print('r2스코어 :', r2)
print('소요시간 :', round(end_time - start_time), '초')

print('================ hist ================')
print(hist)
print('================ hist.history ================')
print(hist.history)
print('================ loss ================')
print(hist.history['loss'])
print('================ val_loss ================')
print(hist.history['val_loss'])
print('==============================================')

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', label='loss', marker='.')
plt.plot(hist.history['val_loss'], c='green', label='val_loss', marker='.')
plt.legend(loc='upper right')
plt.title('diabetes몰라')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.show()