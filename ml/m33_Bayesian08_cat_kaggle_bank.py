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
path = './_data/kaggle/playground-series-s4e1/'

train_csv = pd.read_csv(path + 'train.csv', index_col=[0,1,2])
# print(train_csv) # [165034 rows x 13 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=[0,1,2]) 
# print(test_csv) #  [110023 rows x 12 columns]

submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0) 
# print(submission_csv) # [110023 rows x 1 columns]

# print(train_csv.shape) # (165034, 13)
# print(test_csv.shape) # (110023, 12)
# print(submission_csv.shape) # (110023, 1)

# print(train_csv.columns)
# Index(['CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender', 'Age',
#        'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
#        'EstimatedSalary', 'Exited'],
#       dtype='object')
'''
df = pd.DataFrame(train_csv)

# train_csv = train_csv['Geography'].str.replace('France', '1')
df = df.replace({'Geography':'France'}, '0')
df = df.replace({'Geography':'Germany'}, '1')
df = df.replace({'Geography':'Spain'}, '2')

df = df.replace({'derGen':'Male'}, '1')
df = df.replace({'derGen':'Female'}, '0')
'''
geography_mapping = {'France': 0, 'Germany': 1, 'Spain': 2}
derGen_mapping = {'Male': 1, 'Female': 0}

train_csv['Geography'] = train_csv['Geography'].map(geography_mapping)
train_csv['Gender'] = train_csv['Gender'].map(derGen_mapping)

test_csv['Geography'] = test_csv['Geography'].map(geography_mapping)
test_csv['Gender'] = test_csv['Gender'].map(derGen_mapping)


x = train_csv.drop(['Exited'], axis=1)
y = train_csv['Exited']


random_state=888
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state= random_state, stratify=y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

<<<<<<< HEAD
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

# {'target': 0.8673614687793498, 'params': {'colsample_bytree': 0.5, 'learning_rate': 0.1, 'max_bin': 379.90228087359776, 'max_depth': 10.0, 'min_child_samples': 86.83984485994117, 'min_child_weight': 18.730747028597534, 'num_leaves': 24.03390220808754, 'reg_alpha': 47.30978374655992, 'reg_lambda': 10.0, 'subsample': 1.0}}
# 50 번_걸린시간 :  22.26 초




=======
from catboost import CatBoostClassifier, CatBoostRegressor

#2. 모델

bayesian_params = {
    'learning_rate' : (0.001, 0.1),
    'depth': (2,12),
    'l2_leaf_reg' : (1,10),
    'bagging_temparature' : (0.0, 5.0),
    'border_count' : (32,255),
    'random_strength' : (1,10) 
}
def cat_hamsu(learning_rate, depth,
              l2_leaf_reg, bagging_temparature, 
              border_count, random_strength,
              ):            # 블랙박스 함수라고 함
    params = {
        'iterations' : 200,
        'learning_rate' : learning_rate,
        'depth' : int(round(depth)),
        'l2_leaf_reg' : int(round(l2_leaf_reg)),
        # 'bagging_temparature' : max(min(bagging_temparature,1.0), 0.0),
        'border_count' : int(round(border_count)),
        'random_strength' :int(random_strength),
    }
    
    # cat_features = list(range(x_train.shape[1]))
    
    model = CatBoostRegressor(
                        **params,
                        task_type='GPU',                # GPU 사용 (기본값 : 'CPU')
                        devices='0',                    # 첫번째 CPU 사용 (기본값 : 모든 GPU 사용)
                        early_stopping_rounds=20,      # 조기 종료 (기본값 : None)
                        verbose=0,                     # 매 10번째 반복마다 출력 (기본값 : 100)
                        # eval_metric='F1',             # F1_score를 평가 지표로 설정, 다중분류(macro, micro)는 없음
                        # cat_features=cat_features       # 범주형 feature 지정 
                            )              
    
    model.fit(x_train, y_train,
              eval_set = [(x_test, y_test)],
            #   eval_metric='logloss',
              verbose = 0,
              )
    y_pre = model.predict(x_test)
    result = r2_score(y_test, y_pre)
    
    return result

bay = BayesianOptimization(
    f = cat_hamsu,
    pbounds=bayesian_params,
    random_state=333,
)

n_iter = 100
st = time.time()
bay.maximize(init_points=5, n_iter=n_iter)  # maximize 가 fit이라고 생각
et = time.time()

print(bay.max)
print(n_iter, '번 걸린 시간 :', round(et-st, 2), '초')
>>>>>>> b0025fd (Commit on 2024-07-13)










