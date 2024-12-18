import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split # train test를 사용할때 필요한 sklearn의 모델구성이다.




#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=134, shuffle=True)
'''
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7, #트레인 사이즈의 디폴트 값은 0.75 이다
                                                    # test_size=0.3  테스트 사이즈의 디폴트 값은 0.25 이다
                                                    shuffle=True #셔플의 디폴트는 True다.
                                                    random_state=143, #랜덤 스테이트의 값을 설정하지 않으면 고정된 랜덤값이 아닌 랜덤인 랜덤값이 계속 출력된다.

                                                    )
'''
print('x_train :', x_train)
print('x_test :', x_test)
print('y_train :', y_train)
print('y_test :', y_test)

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가 예측
print('==========================================')
loss = model.evaluate(x_test, y_test)
results = model.predict([11])
print('로스 :', loss)
print('11의 결과값 :', results)

'''
def aaa(a, b):
    a = a+b
    return a
'''

'''
[8 5 9 3 1 7 4]
[10  6  2]
[8 5 9 3 1 7 4]
[10  6  2]
'''
#[검색] train과 test를 섞어서 7:3으로 나누기
#힌트 : 사이킷 런





























