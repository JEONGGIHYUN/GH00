from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from tensorflow.keras.applications import DenseNet201, DenseNet121, DenseNet169
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet, MobileNetV2
from tensorflow.keras.applications import MobileNetV3Small, MobileNetV3Large
from tensorflow.keras.applications import NASNetMobile, NASNetLarge
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB7
from tensorflow.keras.applications import Xception

model_list = [VGG16, VGG19, ResNet50, ResNet50V2, ResNet101, ResNet101V2, ResNet152, ResNet152V2, 
              DenseNet201, DenseNet121, DenseNet169, InceptionV3, InceptionResNetV2,
              MobileNet, MobileNetV2, MobileNetV3Small, MobileNetV3Large, NASNetMobile, NASNetLarge,
              EfficientNetB0, EfficientNetB1, EfficientNetB7
              ]

# 모델별로 결과 출력
for model_class in model_list:
    model = model_class()  # 모델 인스턴스 생성
    model.trainable = False  # 모든 레이어를 훈련 가능하도록 설정
    
    # 모델 정보 출력
    print("==========================================================")
    print("모델명 : ", model.name)
    print("전체 가중치 갯수 : ", len(model.weights))
    print("훈련 가능 갯수 : ", len(model.trainable_weights))
    
    # 메모리 사용량을 줄이기 위해 모델 삭제
    del model

################################## 모델별 결과 출력 ##################################
# for문을 사용해서 결과 밑에 쭉 써라.

# model.trainable = True

# 모델명 :  vgg16
# 전체 가중치 갯수 :  32
# 훈련 가능 갯수 :  32

# 모델명 :  vgg19
# 전체 가중치 갯수 :  38
# 훈련 가능 갯수 :  38

# 모델명 :  resnet50
# 전체 가중치 갯수 :  320
# 훈련 가능 갯수 :  214

# 모델명 :  resnet50v2
# 전체 가중치 갯수 :  272
# 훈련 가능 갯수 :  174

# 모델명 :  resnet101
# 전체 가중치 갯수 :  626
# 훈련 가능 갯수 :  418

# 모델명 :  resnet101v2 
# 전체 가중치 갯수 :  544
# 훈련 가능 갯수 :  344

# 모델명 :  resnet152
# 전체 가중치 갯수 :  932
# 훈련 가능 갯수 :  622

# 모델명 :  resnet152v2
# 전체 가중치 갯수 :  816
# 훈련 가능 갯수 :  514

# 모델명 :  densenet201
# 전체 가중치 갯수 :  1006
# 훈련 가능 갯수 :  604

# 모델명 :  densenet121
# 전체 가중치 갯수 :  606
# 훈련 가능 갯수 :  364

# 모델명 :  densenet169
# 전체 가중치 갯수 :  846
# 훈련 가능 갯수 :  508

# 모델명 :  inception_v3
# 전체 가중치 갯수 :  378
# 훈련 가능 갯수 :  190

# 모델명 :  inception_resnet_v2
# 전체 가중치 갯수 :  898
# 훈련 가능 갯수 :  490

# 모델명 :  mobilenet_1.00_224
# 전체 가중치 갯수 :  137
# 훈련 가능 갯수 :  83

# 모델명 :  mobilenetv2_1.00_224
# 전체 가중치 갯수 :  262
# 훈련 가능 갯수 :  158

# 모델명 :  MobilenetV3small
# 전체 가중치 갯수 :  210
# 훈련 가능 갯수 :  142

# 모델명 :  MobilenetV3large
# 전체 가중치 갯수 :  266
# 훈련 가능 갯수 :  174

# 모델명 :  NASNetMobile
# 전체 가중치 갯수 :  1126
# 훈련 가능 갯수 :  742

# 모델명 :  NASNetLarge
# 전체 가중치 갯수 :  1546
# 훈련 가능 갯수 :  1018

# 모델명 :  efficientnetb0
# 전체 가중치 갯수 :  314
# 훈련 가능 갯수 :  213

# 모델명 :  efficientnetb1
# 전체 가중치 갯수 :  442
# 훈련 가능 갯수 :  301

# 모델명 :  efficientnetb7
# 전체 가중치 갯수 :  1040
# 훈련 가능 갯수 :  711

#########################################################################
# model.trainable = False
# ==========================================================
# 모델명 :  vgg16
# 전체 가중치 갯수 :  32
# 훈련 가능 갯수 :  0
# ==========================================================
# 모델명 :  vgg19
# 전체 가중치 갯수 :  38
# 훈련 가능 갯수 :  0
# ==========================================================
# 모델명 :  resnet50
# 전체 가중치 갯수 :  320
# 훈련 가능 갯수 :  0
# ==========================================================
# 모델명 :  resnet50v2
# 전체 가중치 갯수 :  272
# 훈련 가능 갯수 :  0
# ==========================================================
# 모델명 :  resnet101
# 전체 가중치 갯수 :  626
# 훈련 가능 갯수 :  0
# ==========================================================
# 모델명 :  resnet101v2
# 전체 가중치 갯수 :  544
# 훈련 가능 갯수 :  0
# ==========================================================
# 모델명 :  resnet152
# 전체 가중치 갯수 :  932
# 훈련 가능 갯수 :  0
# ==========================================================
# 모델명 :  resnet152v2
# 전체 가중치 갯수 :  816
# 훈련 가능 갯수 :  0
# ==========================================================
# 모델명 :  densenet201
# 전체 가중치 갯수 :  1006
# 훈련 가능 갯수 :  0
# ==========================================================
# 모델명 :  densenet121
# 전체 가중치 갯수 :  606
# 훈련 가능 갯수 :  0
# ==========================================================
# 모델명 :  densenet169
# 전체 가중치 갯수 :  846
# 훈련 가능 갯수 :  0
# ==========================================================
# 모델명 :  inception_v3
# 전체 가중치 갯수 :  378
# 훈련 가능 갯수 :  0
# ==========================================================
# 모델명 :  inception_resnet_v2
# 전체 가중치 갯수 :  898      
# 훈련 가능 갯수 :  0
# ==========================================================
# 모델명 :  mobilenet_1.00_224
# 전체 가중치 갯수 :  137     
# 훈련 가능 갯수 :  0
# ==========================================================
# 모델명 :  mobilenetv2_1.00_224
# 전체 가중치 갯수 :  262
# 훈련 가능 갯수 :  0
# WARNING:tensorflow:`input_shape` is undefined or non-square, or `rows` is not 224. Weights for input shape (224, 224) will be loaded as the default.
# ==========================================================
# 모델명 :  MobilenetV3small
# 전체 가중치 갯수 :  210
# 훈련 가능 갯수 :  0
# WARNING:tensorflow:`input_shape` is undefined or non-square, or `rows` is not 224. Weights for input shape (224, 224) will be loaded as the default.
# ==========================================================
# 모델명 :  MobilenetV3large
# 전체 가중치 갯수 :  266
# 훈련 가능 갯수 :  0
# ==========================================================
# 모델명 :  NASNetMobile
# 전체 가중치 갯수 :  1126
# 훈련 가능 갯수 :  0
# ==========================================================
# 모델명 :  NASNetLarge
# 전체 가중치 갯수 :  1546
# 훈련 가능 갯수 :  0
# ==========================================================
# 모델명 :  efficientnetb0
# 전체 가중치 갯수 :  314
# 훈련 가능 갯수 :  0
# ==========================================================
# 모델명 :  efficientnetb1
# 전체 가중치 갯수 :  442
# 훈련 가능 갯수 :  0
# ==========================================================
# 모델명 :  efficientnetb7
# 전체 가중치 갯수 :  1040
# 훈련 가능 갯수 :  0