# 01. VGG19
# 02. Xception
# 03. ResNet50
# 04. ResNet101
# 05. InceptionV3
# 06. InceptionResNetV2
# 07. DenseNet121
# 08. MobileNetV2
# 09. NasNetMobile
# 10. EfficeintNetB0

# GAP 써라!!!!!
# 기존거와 최고 성능 비교!!!!

# keras79_all_T_2_cifar100
# keras79_all_T_3_horse
# keras79_all_T_4_rps
# keras79_all_T_5_kaggle_cat_dog
# keras79_all_T_6_men_women

from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.applications import EfficientNetB0

from keras.datasets import cifar10
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, GlobalAveragePooling1D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.callbacks import EarlyStopping
import time

from tensorflow.keras.layers import Resizing

import numpy as np
import tensorflow as tf

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# print("TensorFlow version:", tf.__version__)
# print("GPU Available: ", tf.test.is_built_with_cuda())
# print("GPU Devices: ", tf.config.list_physical_devices('GPU'))

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = tf.image.resize(x_train, (224, 224)).numpy()
x_test = tf.image.resize(x_test, (224, 224)).numpy()

# 데이터 전처리
x_train = x_train/255.
x_test = x_test/255.

# 2. 모델 리스트
models = [ VGG19, Xception, ResNet50, ResNet101, InceptionV3, InceptionResNetV2,
          DenseNet121, MobileNetV2, NASNetMobile, EfficientNetB0
]

# models = [NASNetMobile,] #EfficientNetB0]


# 3. 모델 생성 및 훈련
for i in models:    
        models = i(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)) 
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

# ===========VGG19 모델 테스트===========
# model 이름 :  vgg19
# Loss: 1.2983
# Accuracy: 0.5607
# 훈련시간: 23.82초
# 전체 가중치 수: 238
# 훈련 가능 가중치 수: 4

# model 이름 :  xception
# Loss: 0.8566
# Accuracy: 0.7102
# 훈련시간: 27.13초
# 전체 가중치 수: 322
# 훈련 가능 가중치 수: 4

# model 이름 :  resnet50
# Loss: 2.0436
# Accuracy: 0.2652
# 훈련시간: 32.39초
# 전체 가중치 수: 628   
# 훈련 가능 가중치 수: 4

# ------------------------------------------------
# model 이름 :  vgg19
# Loss: 1.3046
# Accuracy: 0.5421
# 훈련시간: 113.11초
# 전체 가중치 수: 238
# 훈련 가능 가중치 수: 4

# model 이름 :  xception
# Loss: 0.8593
# Accuracy: 0.7028
# 훈련시간: 196.00초
# 전체 가중치 수: 322
# 훈련 가능 가중치 수: 4

# model 이름 :  resnet50
# Loss: 2.3030
# Accuracy: 0.1000
# 훈련시간: 229.73초
# 전체 가중치 수: 628
# 훈련 가능 가중치 수: 4

# model 이름 :  resnet101
# Loss: 2.3035
# Accuracy: 0.1000
# 훈련시간: 304.04초
# 전체 가중치 수: 380
# 훈련 가능 가중치 수: 4

# model 이름 :  inception_v3
# Loss: 1.1890
# Accuracy: 0.6001
# 훈련시간: 252.43초
# 전체 가중치 수: 900
# 훈련 가능 가중치 수: 4

# model 이름 :  inception_resnet_v2
# Loss: 1.0532
# Accuracy: 0.6630
# 훈련시간: 592.40초
# 전체 가중치 수: 900
# 훈련 가능 가중치 수: 4

# 전체 가중치 수: 608
# 훈련 가능 가중치 수: 4
# model 이름 :  densenet121
# Loss: 0.8881
# Accuracy: 0.6926
# 훈련시간: 278.17초

# 전체 가중치 수: 264
# 훈련 가능 가중치 수: 4
# model 이름 :  mobilenetv2_1.00_224
# Loss: 1.1097
# Accuracy: 0.5947
# 훈련시간: 135.78초












