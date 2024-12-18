from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
import matplotlib.pyplot as plt

#1. 데이터
dataset = fetch_california_housing()

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=3, shuffle=True)

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
# * EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor= 'va_loss',
    mode = 'min',
    patience=10,
    restore_best_weights=True
)

start_time = time.time()

model.compile(loss='mse', optimizer='adam')

hist = model.fit(x_train, y_train, epochs=200,
          verbose=1,
          validation_split=0.3)

end_time = time.time()

#4. 평가 예측
loss = model.evaluate(x_test, y_test, verbose=0)

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
plt.rcParams['axes.unicode_minus'] =False #한글 표시 설정

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', label='loss', marker='.')
plt.plot(hist.history['val_loss'], c='green', label='val_loss', marker='.')
plt.legend(loc='upper right')
plt.title('켈리포니아 loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.show()