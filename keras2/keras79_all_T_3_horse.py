import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D,Flatten,GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
import time

train_datagen = ImageDataGenerator(
    rescale=1./255
)
test_datagen = ImageDataGenerator(
    rescale= 1./255
)
# start_time = time.time()
path_train = './_data/image/horse_human/'
path_test = './_data/image/horse_human/'

xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(100, 100),
    batch_size=20000,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True,
)

# xy_test = test_datagen.flow_from_directory(
#     path_test,
#     target_size=(100, 100),
#     batch_size=20000,
#     class_mode='binary',
#     color_mode='rgb',
# )

x_train, x_test, y_train, y_test = train_test_split(xy_train[0][0], xy_train[0][1], train_size=0.7, random_state=3)

# x_train = xy_train[0][0]
# y_train = xy_train[0][1]
# x_test = xy_test[0][0]
# y_test = xy_test[0][1]

# end_time = time.time()

print(x_train.shape,x_test.shape) # (718, 100, 100, 3) (309, 100, 100, 3)
print(y_train.shape,y_test.shape) # (718,) (309,)


from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, GlobalAveragePooling1D

#2. 모델
models = [ VGG19, Xception, ResNet50, ResNet101, InceptionV3, InceptionResNetV2,
          DenseNet121, MobileNetV2, NASNetMobile, EfficientNetB0
]

# models = [NASNetMobile,] #EfficientNetB0]


# 3. 모델 생성 및 훈련
for i in models:    
        models = i(
            weights='imagenet',
            include_top=False,
            input_shape=(100, 100, 3)) 
        models.trainable = False

        # 모델 구성
        model = Sequential()
        model.add(models)
        model.add(GlobalAveragePooling2D()),
        # model.add(GlobalAveragePooling1D()),
        model.add(Dense(32, activation='relu')),
        model.add(Dense(10, activation='softmax'))
       

        # 모델 컴파일
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # 모델 정보 출력
        print("전체 가중치 수:", len(model.weights))
        print("훈련 가능 가중치 수:", len(model.trainable_weights))

        # EarlyStopping 설정
        es = EarlyStopping(
            monitor='val_loss',
            patience=3,
            mode='min',
            restore_best_weights=True
        )

        # 훈련 시작 시간
        start_time = time.time()

        # 모델 훈련
        history = model.fit(
            x_train, y_train,
            batch_size=1,
            epochs=1,
            callbacks=[es],
            verbose=0
        )

        # 훈련 종료 시간
        end_time = time.time()

        # 모델 평가
        loss, acc = model.evaluate(x_test, y_test)
        
        print('model 이름 : ', models.name)
        print(f"Loss: {loss:.4f}")
        print(f"Accuracy: {acc:.4f}")
        print(f"훈련시간: {end_time - start_time:.2f}초")