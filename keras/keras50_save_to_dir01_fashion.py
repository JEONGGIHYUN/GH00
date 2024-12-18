from tensorflow.keras.datasets import fashion_mnist
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
import time
from sklearn.model_selection import train_test_split

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train/255.
y_test = x_test/255.
# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

################ 증폭 #################
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,    # 수평 뒤집기
    vertical_flip=True,      # 수직 뒤집기
    width_shift_range=0.2,   # 평행 이동
    height_shift_range=0.2,  # 평행 이동 수직
    rotation_range=15,        # 정해진 각도만큼 이미지 회전
    # zoom_range=0.4,          # 축소 또는 확대
    # shear_range=0.7,         # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환
    fill_mode='nearest',     # 데이터의 비어있는 곳을 가까운 데이터와 비슷한 값으로 채움 
)

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
    # save_to_dir='C:/TDS/ai5/_data/_save_img/01_fashion/'
).next()[0]
 
print(x_augmented.shape) #(40000, 28, 28, 1)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

print(x_train.shape, x_test.shape)

x_train = np.concatenate((x_train, x_augmented), axis=0)
print(x_train.shape)

y_train = np.concatenate((y_train, y_augmented), axis=0)
print(y_train.shape)



# print(x_train[0].shape) # (28, 28)

# aaa = np.tile(x_train[0], augment_size).reshape(-1, 28, 28, 1),

# print(aaa[0].shape) # (100, 28, 28, 1)

# xy_data = train_datagen.flow(
#     np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1, 28, 28, 1),
#     np.zeros(augment_size),
#     batch_size = augment_size,
#     shuffle=False
# ) #.next()

# print(xy_data)
# print(type(xy_data)) # .next()가 있으면 <class 'tuple'> 없으면 <class 'keras.preprocessing.image.NumpyArrayIterator'>

# print(len(xy_data)) # 1

# # print(xy_data[0].shape) # AttributeError: 'tuple' object has no attribute 'shape'
# # print(xy_data[1].shape) # ValueError: Asked to retrieve element 1, but the Sequence has length 1
# print(xy_data[0][0].shape) # (100, 28, 28, 1)

# # # print(xy_data.shape) # AttributeError: 'tuple' object has no attribute 'shape'
# # print(len(xy_data)) # 2
# # # plt.imshow(x_train[0], cmap='gray')
# # # plt.show()
# # print(xy_data[0].shape) # (100, 28, 28, 1)
# # print(xy_data[1].shape) # (100, )

# plt.figure(figsize=(7,7))
# for i in range(49):
#     plt.subplot(7, 7, i+1)
#     plt.imshow(xy_data[0][0][i], cmap='gray')
    
# plt.show()

#2-1. 함수형 모델
input1 = Input(shape=(28,28,1))
conv2d1 = Conv2D(64,(4,4), strides=2, padding='same')(input1)
max = MaxPooling2D()(conv2d1)
conv2d2 = Conv2D(100, (2,2))(max)
conv2d3 = Conv2D(64, (2,2), activation='relu')(conv2d2)
flt1 =Flatten()(conv2d3)
dense1 = Dense(200, activation='relu')(flt1)
dense2 = Dense(100, activation='relu')(dense1)
dense3 = Dense(50, activation='relu')(dense2)
dense4 = Dense(25, activation='relu')(dense3)
dense5 = Dense(10, activation='softmax')(dense4)
model = Model(inputs=input1, outputs=dense5)


# model.summary()
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
                 validation_split=0.3,
                 callbacks=[es]
                 )

end_time = time.time()

#4. 예측 평가
loss = model.evaluate(x_test, y_test)

y_predict = np.round(model.predict(x_test, 8))

accuracy_score = accuracy_score(y_test, y_predict)

r2 = r2_score(y_test, y_predict)

print('r2_score : ', r2)
print('loss :', loss)
print('acc score :', accuracy_score)
