import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D,Flatten, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time

train_datagen = ImageDataGenerator(
    rescale=1./255
)

path_csv = './_data/kaggle/dogs-vs-cats-redux-Kernels-edition/'

submission_csv = pd.read_csv(path_csv + 'sample_submission.csv', index_col=0)

np_path1 = 'C:/TDS/ai5/_data/catdog set/'

x_train1 = np.load(np_path1 + 'cat_dog_image1.npy')
y_train1 = np.load(np_path1 + 'cat_dog_image2.npy')

np_path2 = 'C:/TDS/ai5/_data/_save_npy/'
x_train2 = np.load(np_path2 + 'x_train2.npy')

y_train2 = np.load(np_path2 + 'y_train2.npy')

x_test = np.load(np_path2 + 'x_test2.npy')

y_test = np.load(np_path2 + 'y_test2.npy')

x_train = np.concatenate((x_train1, x_train2), axis=0)

y_train = np.concatenate((y_train1, y_train2), axis=0)

augment_size = 10

randidx = np.random.randint(x_train.shape[0], size=augment_size) # 60000, size=40000

x_augmented = x_train[randidx].copy()#  메모리 안전 

y_augmented = y_train[randidx].copy()
# x_augmented = x_augmented.reshape()

x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size = augment_size,
    shuffle=False,
).next()[0]
 
# x_train = x_train.reshape(19997, 100, 100, 3)
# y_train = y_train.reshape(5000, 100, 100, 3)

x_train = np.concatenate((x_train, x_augmented), axis=0)

y_train = np.concatenate((y_train, y_augmented), axis=0)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(40,(2,2), input_shape=(80,80,3),padding='same',strides=2))
model.add(Conv2D(55, (2,2), activation='relu',strides=1, padding='same'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(55, (2,2), activation='relu',strides=2, padding='same'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(60, (2,2), activation='relu',strides=2, padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) # accuracy, mse

start_time = time.time()
################
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
################
es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=5, verbose=1,
                   restore_best_weights=True)
################
import datetime

date = datetime.datetime.now()

date = date.strftime('%m%d_%H%M')

filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path_csv, 'k42_2_', date,'_', filename])

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode = 'auto',
    verbose= 1,
    save_best_only=True,
    filepath = filepath
)


model.fit(x_train, y_train, epochs=1,  batch_size=1,
                 verbose=1,
                 validation_split=0.3,
                 callbacks=[es]
                 )

end_time = time.time()

#4. 평가 예측
loss = model.evaluate(x_test, y_test,batch_size=1)

y_submit = model.predict(x_test,batch_size=1)
# y_pred = model.predict(x_test,batch_size=1)
# print(y_pr

# y_pred = np.round(y_pred)

# accuracy_score = accuracy_score(y_test, y_pred)

submission_csv['label'] = (y_submit)

submission_csv.to_csv(path_csv + 'submission_08_06_.csv')

# print('로스 :', loss)
# print('acc_score :', accuracy_score)
# print('소요시간 :', round(end_time - start_time), '초')



































































