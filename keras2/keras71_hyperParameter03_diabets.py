import numpy as np
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
x, y = load_diabetes(return_X_y=True)

x_train,x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=136, shuffle=True,)# stratify=y)

print(x_train.shape, y_train.shape)

#2. 모델
def build_model(drop=0.5, optimizer='Adagrad', activation='relu',
                node1=128, node2=64, node3=32, node4=16, node5=8, lr=0.001):
    inputs = Input(shape=(10, ), name='inputs')
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
                           n_iter=10,
                           verbose=1,
                        #    n_jobs=-1
                           )
import time
start = time.time()
model.fit(x_train,y_train, epochs=100)
end = time.time()

print('걸린시간 : ', round(end - start, 2))
print('model.best_params_', model.best_params_)
print('model.best_estimator_', model.best_estimator_)
print('model.best_score_', model.best_score_)
print('model.score', model.score(x_test,y_test))












