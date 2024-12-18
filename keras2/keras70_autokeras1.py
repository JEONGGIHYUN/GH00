import autokeras as ak
from autokeras import ImageClassifier
import tensorflow as tf
from tensorflow.keras.datasets import mnist

print(ak.__version__)
print(tf.__version__)

import time
#1. 데이터

(x_train,x_test),(y_train,y_test) = mnist.load_data()

print(x_train.shape, x_test.shape)

#2 모델
model = ImageClassifier(
    overwrite=False,
    max_trials=1
)

#3. 컴파일 훈련
start_time = time.time()
model.fit(x_train,y_train,epochs=1,validation_split=0.15)
end_time = time.time()

### 최적의 출력 모델 ######
best_model = model.export_model()
print(best_model.summary())

### 최적의 모델 저장 ###
path = 'C:/ai5/_save/autokeras/'

best_model.save(path + 'keras70_autokeras1.h5')

#4. 평가 예측
y_pred = model.predict(x_test)
results = model.evaluate(x_test,y_pred)
print('model 결과 :', results)

y_pred2 = best_model.predict(x_test)
results2 = best_model.evaluate(x_test,y_pred2)
print('model 결과 :', results2)

print('걸린시간 :',round(end_time - start_time, 2), '초')


































































