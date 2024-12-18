import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier,XGBRegressor
from sklearn.metrics import accuracy_score, r2_score
from bayes_opt import BayesianOptimization
import time
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

#1. 데이터
path = 'C:/ai5/_data/dacon/따릉이/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
print(train_csv) # [1459 rows x 10 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=0) 
print(test_csv) # [715 rows x 9 columns]

submission_csv = pd.read_csv(path + 'submission.csv', index_col=0) # [715 rows x 1 columns]

############ 결측치 처리 1. 삭제 ##############
train_csv.isnull().sum()

train_csv = train_csv.dropna() 

test_csv = test_csv.fillna(test_csv.mean()) # 결측치 채우기 



x = train_csv.drop(['count'], axis=1)

y = train_csv['count']

# x.boxplot() # hour_bef_visibility
# x.plot.box()
# plt.show()

x['hour_bef_visibility'] = np.log1p(x['hour_bef_visibility']) # 지수변환 np.exp1m
random_state=888
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state= random_state)

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

    model = XGBRegressor(**params,)
    
    model.fit(x_train, y_train,
              eval_set=[(x_test, y_test)],
            #   eval_metric='logloss', 
              verbose=0,)
    
    y_predict = model.predict(x_test)
    results = r2_score(y_test, y_predict)
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

# {'target': 0.7929755733353221, 'params': {'colsample_bytree': 1.0, 'learning_rate': 0.1, 'max_bin': 453.6426094719462, 'max_depth': 6.356689364055212, 'min_child_samples': 36.66676083505085, 'min_child_weight': 20.665115682527425, 'num_leaves': 27.936355050919552, 'reg_alpha': 12.050346161196096, 'reg_lambda': 6.6966369497885045, 'subsample': 1.0}}
# 50 번_걸린시간 :  7.56 초







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







