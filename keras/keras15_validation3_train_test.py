import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


x = np.array(range(1,17))
y = np.array(range(1,17))

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=144, train_size=0.8, shuffle=True)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, random_state=134, train_size=0.5, shuffle=True)

print(x_train)
print(x_test)
print(x_val)

print(y_train)
print(y_test)
print(y_val)



#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=5,
          verbose=1
<<<<<<< HEAD
          ,validation_data=(x_val,y_val)) # verbose의 디폴트 값은 1 이다.



# verbose0 : 침묵
# verbose1 : 디폴트
# verbose2 : 프로그래스바 삭제
# verbose4 : 에포만 나온다
=======
          ,validation_data=(x_val,y_val)) 
>>>>>>> b0025fd (Commit on 2024-07-13)

#4. 평가 예측
print('==========================================')
loss = model.evaluate(x_test, y_test)
results = model.predict([11])
print('로스 :', loss)
print('11의 결과값 :', results)
