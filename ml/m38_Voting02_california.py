import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestRegressor, VotingRegressor,VotingClassifier
from sklearn.linear_model import LogisticRegression #(로지스틱 리그레서는 회귀가 아닌 분류이다.)
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier

#1. 데이터
x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1367)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
xgb = XGBRegressor()
rf = RandomForestRegressor()
cat = CatBoostRegressor()

# model = XGBRegressor()
model = VotingRegressor(
    estimators= [('xgb', xgb), ('rf', rf), ('cat', cat)]
)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가 예측
results = model.score(x_test, y_test)
print('예측 점수 :', results )

y_pre = model.predict(x_test)
results = r2_score(y_test, y_pre)
print('results_score : ', results)

# XGB
# 예측 점수 : 0.8259431875142139
# results_score :  0.8259431875142139

# Voting
# 예측 점수 : 0.8376068386425088
# results_score :  0.8376068386425088