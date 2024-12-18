import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPooling2D, Input
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#1. 데이터

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape) # (10000, 32, 32, 3) (10000, 1)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip=True,    # 수평 뒤집기
    # vertical_flip=True,      # 수직 뒤집기
    # width_shift_range=0.2,   # 평행 이동
    # height_shift_range=0.2,  # 평행 이동 수직
    # rotation_range=15,        # 정해진 각도만큼 이미지 회전
    # zoom_range=0.4,          # 축소 또는 확대
    # shear_range=0.7,         # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환
    # fill_mode='nearest',     # 데이터의 비어있는 곳을 가까운 데이터와 비슷한 값으로 채움 
)


#1-1 스케일링
x_train = x_train / 255.
x_test = x_test / 255.

#1-2 원핫 인코딩
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape, y_test.shape) # (50000, 100) (10000, 100)

augment_size = 50000

randidx = np.random.randint(x_train.shape[0], size=augment_size) # 60000, size=40000

x_augmented = x_train[randidx].copy()#  메모리 안전 

y_augmented = y_train[randidx].copy()
# x_augmented = x_augmented.reshape()

x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size = augment_size,
    shuffle=False,
).next()[0]
 
x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)

x_train = np.concatenate((x_train, x_augmented), axis=0)

y_train = np.concatenate((y_train, y_augmented), axis=0)


#2-1. 함수형 모델
input1 = Input(shape=(32,32,3))
conv2d1 = Conv2D(100,(2,2), strides=2, padding='same')(input1)
max = MaxPooling2D()(conv2d1)
conv2d2 = Conv2D(50, (1,1), activation='relu')(max)
flt1 =Flatten()(conv2d2)
dense1 = Dense(500, activation='relu')(flt1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(500, activation='relu')(drop1)
dense3 = Dense(300, activation='relu')(dense2)
dense4 = Dense(100, activation='softmax')(dense3)
model = Model(inputs=input1, outputs=dense4)


#3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

start_time = time.time()

es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=10, verbose=1,
                   restore_best_weights=True)

model.fit(x_train, y_train, epochs=100,  batch_size=546,
                 verbose=2,
                 validation_split=0.2,
                 callbacks=[es]
                 )

end_time = time.time()

#4. 예측 평가
loss = model.evaluate(x_test, y_test)
y_pred = np.round(model.predict(x_test))

accuracy_score = accuracy_score(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

print('r2_score : ', r2)
print('loss :', loss)
print('acc score :', accuracy_score)
