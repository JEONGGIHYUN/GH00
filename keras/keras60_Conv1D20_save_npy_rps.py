import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D,Flatten, Conv1D, Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time

train_datagen = ImageDataGenerator(
    rescale=1./255
)
test_datagen = ImageDataGenerator(
    rescale= 1./255
)
# start_time = time.time()
path_train = 'C:/TDS/ai5/_data/image/rps/'
path_test = 'C:/TDS/ai5/_data/image/rps/'

xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(100, 100),
    batch_size=20000,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True,
)

np_path = 'C:/TDS/ai5/_data/_save_npy/'
np.save(np_path + 'keras45_03_x_train.npy', arr=xy_train[0][0])
np.save(np_path + 'keras45_03_y_train.npy', arr=xy_train[0][1])

x_train, x_test, y_train, y_test = train_test_split(xy_train[0][0], xy_train[0][1], train_size=0.7, random_state=3)

# # x_train = xy_train[0][0]
# # y_train = xy_train[0][1]
# # x_test = xy_test[0][0]
# # y_test = xy_test[0][1]

# # end_time = time.time()

# print(x_train.shape,x_test.shape) # (1764, 100, 100, 3) (756, 100, 100, 3)
# print(y_train.shape,y_test.shape) # (1764, 3) (756, 3)


#2. 모델 구성
model = Sequential()
model.add(Conv2D(5,(4,4), input_shape=(100,100,3)))
model.add(Conv2D(5, (4,4), activation='relu'))
model.add(MaxPooling2D(4,4))
model.add(Reshape(target_shape=(529,5)))
model.add(Conv1D(filters=10,kernel_size=2,input_shape=(529,5)))
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
print(y_pred)

y_pred = np.round(y_pred)

accuracy_score = accuracy_score(y_test, y_pred)


print('로스 :', loss)
print('acc_score :', accuracy_score)
print('소요시간 :', round(end_time - start_time), '초')

# Conv1D
# 로스 : [0.007903918623924255, 0.9986772537231445]
# acc_score : 0.9986772486772487
# 소요시간 : 5 초