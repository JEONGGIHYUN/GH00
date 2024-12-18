import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D,Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time

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
test_datagen = ImageDataGenerator(
    rescale= 1./255
)
# start_time = time.time()
path_train = './_data/image/rps/'
path_test = './_data/image/rps/'

xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(100, 100),
    batch_size=20000,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True,
)

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(100, 100),
    batch_size=20000,
    class_mode='categorical',
    color_mode='rgb',
)

# x_train, x_test, y_train, y_test = train_test_split(xy_train[0][0], xy_train[0][1], train_size=0.7, random_state=3)

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

# end_time = time.time()

# print(x_train.shape,x_test.shape) # (2520, 100, 100, 3) (2520, 100, 100, 3)
# print(y_train.shape,y_test.shape) # (2520, 3) (2520,)

augment_size = 3000

randidx = np.random.randint(x_train.shape[0], size=augment_size) # 60000, size=40000

x_augmented = x_train[randidx].copy()#  메모리 안전 

y_augmented = y_train[randidx].copy()
# x_augmented = x_augmented.reshape()

x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size = augment_size,
    shuffle=False,
    # save_to_dir='C:/TDS/ai5/_data/_save_img/07_rps/'
).next()[0]
 
# # x_train = x_train.reshape(19997, 100, 100, 3)
# # y_train = y_train.reshape(2520,3)

# x_train = np.concatenate((x_train, x_augmented), axis=0)

# y_train = np.concatenate((y_train, y_augmented), axis=0)


#2. 모델 구성
model = Sequential()
model.add(Conv2D(5,(4,4), input_shape=(100,100,3)))
model.add(Conv2D(5, (4,4), activation='relu'))
model.add(MaxPooling2D(4,4))
model.add(Flatten())
model.add(Dense(30, activation='relu'))
model.add(Dense(3, activation='softmax'))

#3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) # accuracy, mse

start_time = time.time()
################
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
################
es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=3, verbose=1,
                   restore_best_weights=True)
################

model.fit(x_train, y_train, epochs=30,  batch_size=100,
                 verbose=1,
                 validation_split=0.3,
                 callbacks=[es]
                 )

end_time = time.time()

#4. 평가 예측
loss = model.evaluate(x_test, y_test)


y_pred = model.predict(x_test)
# print(y_pred)

y_pred = np.round(y_pred)

accuracy_score = accuracy_score(y_test, y_pred)


print('로스 :', loss)
print('acc_score :', accuracy_score)
print('소요시간 :', round(end_time - start_time), '초')

# 로스 : [1.0458170175552368, 0.3464285731315613]
# acc_score : 0.1595238095238095
# 소요시간 : 4 초

# 로스 : [0.8687019348144531, 0.5726190209388733]
# acc_score : 0.5214285714285715
# 소요시간 : 4 초