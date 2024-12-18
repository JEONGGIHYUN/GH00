import numpy as np
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score
from bayes_opt import BayesianOptimization
from catboost import CatBoostRegressor
import time
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x, y = fetch_california_housing(return_X_y=True)

random_state=888
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state= random_state)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
parameters = {
    'learning_rate' : [0.01, 0.03, 0.05, 0.1, 0.2],
    'depth' : [4, 6, 7, 10, 12],
    'l2_leaf_reg' :  [1,3,5,7,10],
    'bagging_temperature' : [0.0, 0.5, 1.0, 2.0, 5.0],
    'border_count' : [32, 64, 128,  256],
    'random_strength' : [1, 5, 10],
}


def xgb_hamsu(learning_rate, depth, l2_leaf_reg, bagging_temperature, border_count, random_strength):
    params = {
        'n_estimators' : 100,
        'learning_rate' : learning_rate,
        'depth' : int(round(depth)), # 무조건 정수형
        'l2_leaf_reg' :  int(round(l2_leaf_reg)),
        'bagging_temperature' : bagging_temperature,
        'border_count' : int(round(border_count)),
        'random_strength' : int(round(random_strength))
    }

    model = CatBoostRegressor(**params,)
    
    model.fit(x_train, y_train,
              eval_set=[(x_test, y_test)],
            #   eval_metric='logloss', 
              verbose=0,)
    
    y_predict = model.predict(x_test)
    results = r2_score(y_test, y_predict)
    return results

bay = BayesianOptimization(
    f=xgb_hamsu,
    pbounds=parameters,
    random_state=3333,
)


n_iter=50
start = time.time()
bay.maximize(init_points=5, n_iter=n_iter)
end = time.time()
# optimizer.maximize(init_points=5,
#                    n_iter=20)
# print(optimizer.)

print(bay.max)
print(n_iter, '번_걸린시간 : ', round(end - start, 2),'초')

# {'target': 0.8349522988671683, 'params': {'colsample_bytree': 1.0, 'learning_rate': 0.1, 'max_bin': 442.32630090780845, 'max_depth': 10.0, 'min_child_samples': 47.5813240602489, 'min_child_weight': 22.734594931446605, 'num_leaves': 32.88307294597977, 'reg_alpha': 5.355398026173484, 'reg_lambda': 0.6203288326357366, 'subsample': 0.7993640691698312}}
# 50 번_걸린시간 :  13.42 초














