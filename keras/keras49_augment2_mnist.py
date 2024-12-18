import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
import time
from sklearn.model_selection import train_test_split

(x_train, y_train), (x_test, y_test) = mnist.load_data()


# print(x_train)
# print(x_train[0])

# print(y_train[0])


print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

################ 증폭 #################
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





#######스케일링
x_train = x_train / 255.
x_test = x_test / 255.
print(np.max(x_train),np.min(x_train))

########스케일링 1-2
# x_train=(x_train - 127.5) / 127.5
# x_test=(x_test - 127.5) / 127.5

########스케일링 2. MinMaxScaler
# scaler = MinMaxScaler()

### 원핫 케라스
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# print(np.max(x_train), np.min(x_train))


### 원핫 판다스
# y_train = pd.get_dummies(y_train)
# y_test =  pd.get_dummies(y_test)

augment_size = 40000

print(x_train.shape[0])

randidx = np.random.randint(x_train.shape[0], size=augment_size) # 60000, size=40000
print(randidx) # [25398 39489  6129 ... 52641 22598 45654] 랜덤 생성 4만개 

print(np.min(randidx),np.max(randidx)) # 0 59999

print(x_train[0].shape) # (28, 28)

x_augmented = x_train[randidx].copy()#  메모리 안전 

y_augmented = y_train[randidx].copy()

print(x_augmented.shape,y_augmented.shape) # (40000, 28, 28) (40000,)

x_augmented = x_augmented.reshape(
    x_augmented.shape[0], # 40000
    x_augmented.shape[1], #28
    x_augmented.shape[2], 1 #28,1
)

x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size = augment_size,
    shuffle=False,
).next()[0]
 
print(x_augmented.shape) #(40000, 28, 28, 1)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

print(x_train.shape, x_test.shape)

x_train = np.concatenate((x_train, x_augmented), axis=0)
print(x_train.shape)

y_train = np.concatenate((y_train, y_augmented), axis=0)
print(y_train.shape)

#2. 모델
# model = Sequential()
# model.add(Conv2D(64, (2,2), input_shape=(28, 28, 1),
#                  strides=2,
#                  padding='same')) # 27, 27 , 10
# model.add(MaxPooling2D())
#                        #shape = (batch_sitze, rows, culumes, channels)
# model.add(Conv2D(filters=64, kernel_size=(3, 3))) # 25, 25, 20
#                        #shape = (batch_size, heights, widths, channerls)
# model.add(Conv2D(32, (2, 2), activation='relu')) #24, 24, 32
# model.add(Flatten())
# model.add(Dense(units=32, activation='relu'))
# model.add(Dense(units=116, input_shape=(32,)))
#                         # shape = (batch_size, input_dim)

# model.add(Dense(10, activation='softmax'))

#2-1. 함수형 모델
input1 = Input(shape=(28,28,1))
conv2d1 = Conv2D(64,(2,2), strides=2, padding='same')(input1)
max = MaxPooling2D()(conv2d1)
conv2d2 = Conv2D(64, (3,3))(max)
conv2d3 = Conv2D(32, (2,2), activation='relu')(conv2d2)
flt1 =Flatten()(conv2d3)
dense1 = Dense(32, activation='relu')(flt1)
dense2 = Dense(116)(dense1)
dense3 = Dense(10, activation='softmax')(dense2)
model = Model(inputs=input1, outputs=dense3)

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
# ################파일 명 만 들 기
# import datetime
# date = datetime.datetime.now()
# print(date)
# print(type(date)) # <class 'datetime.datetime'>
# date = date.strftime('%m%d_%H%M')
# print(date)
# print(type(date))

# path = 'C:/TDS/ai5/study/_save/keras35_/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
# filepath = ''.join([path, 'k35_04_', date, '_', filename])
# ################
# ################
# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode = 'auto',
#     verbose= 1,
#     save_best_only=True,
#     filepath = filepath
# )
# ################
model.fit(x_train, y_train, epochs=100,  batch_size=3350,
                 verbose=2,
                 validation_split=0.3,
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