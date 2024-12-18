import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# print(x_train)
# print(x_train[0])

# print(y_train[0])


print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28 * 28)
x_test = x_test.reshape(10000, 28 * 28)



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

#2. 모델
def build_model(drop=0.5, optimizer='Adagrad', activation='relu',
                node1=128, node2=64, node3=32, node4=16, node5=8, lr=0.001):
    inputs = Input(shape=(784, ), name='inputs')
    x = Dense(node1, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)    
    x = Dense(node4, activation=activation, name='hidden4')(x)
    x = Dense(node5, activation=activation, name='hidden5')(x)
    outputs = Dense(1, activation='linear', name='outputs')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=optimizer, metrics=['mae'],
                  loss='mae')
    return model

def create_hyperparameter():
    batchs = [100,200,300,400,500]
    optimizers = ['adam', 'adagred', 'rmsprop', 'adadelta']
    dropouts = [0.2, 0.3, 0.4, 0.5]
    activations = ['relu', 'elu', 'selu', 'linear']
    node1 = [128, 64, 32, 7]
    node2 = [128, 40, 4, 2]    
    node3 = [128, 64, 32, 7]
    node4 = [128, 65, 32, 8]
    node5 = [128, 70, 16, 5]
    return {'batch_size' : batchs,
            'optimizer' : optimizers,
            'drop' : dropouts,
            'activation' : activations,
            'node1' : node1,
            'node2' : node2,
            'node3' : node3,
            'node4' : node4,
            'node5' : node5,
            }

hyperparameters = create_hyperparameter()
print(hyperparameters)

from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

keras_model = KerasRegressor(build_fn=build_model, verbose=1)

model = RandomizedSearchCV(keras_model, hyperparameters, cv=5,
                           n_iter=5,
                           verbose=1,
                        #    n_jobs=-1
                           )
import time
start = time.time()
model.fit(x_train,y_train, epochs=50)
end = time.time()

print('걸린시간 : ', round(end - start, 2))
print('model.best_params_', model.best_params_)
print('model.best_estimator_', model.best_estimator_)
print('model.best_score_', model.best_score_)
print('model.score', model.score(x_test,y_test))






