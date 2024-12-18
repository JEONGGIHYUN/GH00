from sklearn.datasets import load_digits
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

x, y = load_digits(return_X_y=True) # 사이킷런에서 사용 가능한 방식이다.

# print(x)
# print(y)
# print(x.shape, y.shape) #(1797, 64) (1797,)

print(pd.value_counts(y, sort=False))
# 0    178
# 1    182
# 2    177
# 3    183
# 4    181
# 5    182
# 6    181
# 7    179
# 8    174
# 9    180
# dtype: int64

y = pd.get_dummies(y)
print(y)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1346, train_size=0.75, shuffle=True, stratify=y)

###############################################
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)
################################################
# print(x_train)
print(np.min(x_train), np.max(x_train)) # 0.0 16.0
print(np.min(x_test), np.max(x_test)) # 0.0 16.0


#2. 모델 구성
model = Sequential ()
model.add(Dense(500, input_dim=64, activation='relu'))
model.add(Dense(250))
model.add(Dense(125))
model.add(Dense(75))
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(10))
model.add(Dense(10, activation='softmax'))

#3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
start_time = time.time()
es = EarlyStopping(
    monitor='val_loss',
    mode = 'min',
    patience=10,
    restore_best_weights=True
)

model.fit(x_train, y_train, epochs=3000, batch_size=10000, validation_split=0.2)

end_time = time.time()

#4. 예측 평가
loss = model.evaluate(x_test, y_test)
y_predict = np.around(model.predict(x_test))
accuracy_score = accuracy_score(y_test, y_predict)

print('loss :', loss)
print('acc score :', accuracy_score)
print('time :', round(end_time - start_time, 2), '초')

'''
loss : [0.16231749951839447, 0.9733333587646484]
acc score : 0.9733333333333334
time : 49.37 초

loss : [0.07097781449556351, 0.9777777791023254]
acc score : 0.9777777777777777
time : 49.53 초

'''