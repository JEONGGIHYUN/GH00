import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score
from bayes_opt import BayesianOptimization
import time
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

#1. 데이터
path = './_data/dacon/diabetes/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
# print(train_csv) # [652 rows x 9 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=0)
# print(test_csv) # [116 rows x 8 columns]

submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)
# print(submission_csv)

# print(train_csv.shape) # (652, 9)
# print(test_csv.shape) # (116, 8)
# print(submission_csv.shape) # (116, 1)

# print(train_csv.columns)
# Index(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
    #    'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'],
    #   dtype='object')

x = train_csv.drop(['Outcome'], axis=1)

y = train_csv['Outcome']

random_state=888
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state= random_state, stratify=y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
bayesian_params = {
    'learning_rate' : (0.001, 0.1),
    'max_depth' : (3, 10),
    'num_leaves' : (24, 40),
    'min_child_samples' : (10, 200),
    'min_child_weight' : (1, 50),
    'subsample' : (0.5, 1),
    'colsample_bytree' : (0.5, 1),
    'max_bin' : (9, 500),
    'reg_lambda' : (-0.001, 10),
    'reg_alpha' : (0.01, 50)
}

def xgb_hamsu(learning_rate, max_depth, num_leaves, min_child_samples, min_child_weight, subsample, colsample_bytree, max_bin, reg_lambda, reg_alpha):
    params = {
        'n_estimators' : 100,
        'learning_rate' : learning_rate,
        'max_depth' : int(round(max_depth)), # 무조건 정수형
        'num_leaves' : int(round(num_leaves)),
        'min_child_samples' : int(round(min_child_samples)),
        'min_child_weight' : int(round(min_child_weight)),
        'subsample' : max(min(subsample, 1), 0),
        'colsample_bytree' : colsample_bytree,
        'max_bin' : max(int(round(max_bin)), 10),
        'reg_lambda' : max(reg_lambda, 0),
        'reg_alpha'  : reg_alpha,
    }

    model = XGBClassifier(**params,)
    
    model.fit(x_train, y_train,
              eval_set=[(x_test, y_test)],
            #   eval_metric='logloss', 
              verbose=0,)
    
    y_predict = model.predict(x_test)
    results = accuracy_score(y_test, y_predict)
    return results

bay = BayesianOptimization(
    f=xgb_hamsu,
    pbounds=bayesian_params,
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

# {'target': 0.7938931297709924, 'params': {'colsample_bytree': 1.0, 'learning_rate': 0.1, 'max_bin': 233.60152060658584, 'max_depth': 3.184321605578451, 'min_child_samples': 128.04527002868784, 'min_child_weight': 5.584920346062064, 'num_leaves': 25.572622932538714, 'reg_alpha': 15.233182700432417, 'reg_lambda': 6.010489137336423, 'subsample': 1.0}}
# 50 번_걸린시간 :  5.73 초














