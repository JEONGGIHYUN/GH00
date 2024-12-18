import numpy as np
from sklearn.datasets import load_breast_cancer,load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier,XGBRegressor
from sklearn.metrics import accuracy_score,r2_score
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=213,shuffle=True)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

import xgboost as xgb
early_stop = xgb.callback.EarlyStopping(
    rounds=50,
    metric_name='logloss', #이진은 logloss, 다중은 mlogloss
    data_name='validation_0',
    save_best=True
)

#2. 모델
model = XGBRegressor(
    n_estimators = 500,
    max_depth = 8,
    gamma = 0,
    min_child_weight = 0,
    subsample=0.4,
    reg_alpha=0,
    reg_lambda=1,
    # callbacks=[early_stop],
    random_state=2146,
    eval_metric='logloss', # 2.1.1 버전에서 여기서 써야 한다.
)

#3. 훈련
model.fit(x_train, y_train,
          eval_set=[(x_test, y_test)],

        #   eval_metric='mlogloss',
          verbose=1,
          )

#4. 평가 예측
results = model.score(x_test, y_test)
print('최종점수 :', results)

y_pred = model.predict(x_test)
acc = r2_score(y_test, y_pred)
print('acc:', acc)

print(model.feature_importances_)

# 최종점수 : 0.956140350877193
# acc: 0.956140350877193
# [0.00657149 0.02832544 0.0194451  0.01801375 0.01347957 0.00306432
#  0.01568072 0.17670521 0.00647213 0.03701359 0.01123884 0.00238928
#  0.02764063 0.03286134 0.0171529  0.02342392 0.00506255 0.00449685
#  0.01343773 0.01034998 0.01359118 0.0471011  0.26093343 0.05805217
#  0.01218688 0.00875304 0.01373413 0.09908672 0.00699362 0.00674231]

thresholds = np.sort(model.feature_importances_) # 오름차순
# print(thresholds)

# [0.00238928 0.00306432 0.00449685 0.00506255 0.00647213 0.00657149
#  0.00674231 0.00699362 0.00875304 0.01034998 0.01123884 0.01218688
#  0.01343773 0.01347957 0.01359118 0.01373413 0.01568072 0.0171529
#  0.01801375 0.0194451  0.02342392 0.02764063 0.02832544 0.03286134
#  0.03701359 0.0471011  0.05805217 0.09908672 0.17670521 0.26093343]


from sklearn.feature_selection import SelectFromModel

for i in thresholds:
    selection = SelectFromModel(model, threshold = 0.1, prefit = True)
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    
    select_model = XGBRegressor(  n_estimators = 10,
                                    max_depth = 8,
                                    gamma = 0,
                                    min_child_weight= 0,
                                    subsample= 0.4,
                                    # reg_alpha= 0,    # L1 규제 리소
                                    # reg_lambda= 1,   # L2 규제 라지
                                    # eval_metric='mlogloss', # 2.1.1 버전에서 컴파일이 아니라 모델로 가야됨.. / 이진분류에서error도 쓸 수 있음.
                                    # callbacks=[early_stop], 
                                    random_state=3377,)
    
    select_model.fit(select_x_train, y_train,
                     eval_set=[(select_x_test, y_test)],
                     verbose=0,
                     )
    select_y_predict = select_model.predict(select_x_test)
    score = r2_score(y_test, select_y_predict)
    
    print('Trech=%.3f, n=%d, ACC:%.2f%%' %(i, select_x_train.shape[1], score*100))














