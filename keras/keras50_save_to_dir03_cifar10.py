import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Input
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#1. 데이터

(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# print(x_train)
# print(x_train[0])

# print(y_train[0])
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,    # 수평 뒤집기
    vertical_flip=True,      # 수직 뒤집기
    width_shift_range=0.2,   # 평행 이동
    height_shift_range=0.2,  # 평행 이동 수직
    rotation_range=15,        # 정해진 각도만큼 이미지 회전
    zoom_range=0.4,          # 축소 또는 확대
    shear_range=0.7,         # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환
    fill_mode='nearest',     # 데이터의 비어있는 곳을 가까운 데이터와 비슷한 값으로 채움 
)

# print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape) # (10000, 32, 32, 3) (10000, 1)

#1-1 스케일링
x_train = x_train / 255.
x_test = x_test / 255.
# print(np.max(x_train),np.min(x_train)



#1-2 원핫 인코딩
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# augment_size = 50000

# print(x_train.shape[0])

# randidx = np.random.randint(x_train.shape[0], size=augment_size) # 60000, size=40000
# print(randidx) # [25398 39489  6129 ... 52641 22598 45654] 랜덤 생성 4만개 

# print(np.min(randidx),np.max(randidx)) # 0 59999

# print(x_train[0].shape) # (28, 28)

# x_augmented = x_train[randidx].copy()#  메모리 안전 

# y_augmented = y_train[randidx].copy()

# print(x_augmented.shape,y_augmented.shape) # (40000, 28, 28) (40000,)

# # x_augmented = x_augmented.reshape()

# x_augmented = train_datagen.flow(
#     x_augmented, y_augmented,
#     batch_size = augment_size,
#     shuffle=False,
#     # save_to_dir='C:/TDS/ai5/_data/_save_img/03_cifar10/'
# ).next()[0]
 
# print(x_augmented.shape) #(40000, 28, 28, 1)

# # x_train = x_train.reshape(50000, 32, 32, 3)
# # x_test = x_test.reshape(10000, 32, 32, 3)

# # print(x_train.shape, x_test.shape)

# x_train = np.concatenate((x_train, x_augmented), axis=0)
# # print(x_train.shape)

# y_train = np.concatenate((y_train, y_augmented), axis=0)
# print(y_train.shape)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, random_state=3, train_size=0.7)

#2-1. 함수형 모델
input1 = Input(shape=(32,32,3))
conv2d1 = Conv2D(32,(4,4), strides=1, padding='same')(input1)
max = MaxPooling2D(2,2)(conv2d1)
conv2d2 = Conv2D(100, (2,2))(max)
conv2d3 = Conv2D(50, (1,1), activation='relu')(conv2d2)
conv2d4 = Conv2D(30, (1,1), activation='relu')(conv2d3)
flt1 =Flatten()(conv2d4)
dense1 = Dense(150, activation='relu')(flt1)
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(100, activation='relu')(drop1)
dense3 = Dense(50, activation='relu')(dense2)
dense4 = Dense(10, activation='softmax')(dense3)
model = Model(inputs=input1, outputs=dense4)

#3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

start_time = time.time()

################
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
################
es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=10, verbose=1,
                   restore_best_weights=True)
################
model.fit(x_train, y_train, epochs=100,  batch_size=3350,
                 verbose=2,
                 validation_split=0.2,
                 callbacks=[es]
                 )

end_time = time.time()

#4. 예측 평가
loss = model.evaluate(x_test, y_test)

y_predict = np.round(model.predict(x_test))

accuracy_score = accuracy_score(y_test, y_predict)

r2 = r2_score(y_test, y_predict)

print('r2_score : ', r2)
print('loss :', loss)
print('acc score :', accuracy_score)

# r2_score :  0.2763332924459934
# loss : [1.0421723127365112, 0.635699987411499]
# acc score : 0.545

# r2_score :  0.05428581496709327
# loss : [1.7161529064178467, 0.3526333272457123]
# acc score : 0.2417

# r2_score :  0.06950345450061698
# loss : [1.6879459619522095, 0.3605000078678131]
# acc score : 0.2544666666666667