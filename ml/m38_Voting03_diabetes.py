import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression #(로지스틱 리그레서는 회귀가 아닌 분류이다.)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

#1. 데이터
datasets = load_diabetes()

x = datasets.data
y = datasets.target

# print(x.shape, y.shape) (442, 10) (442, )

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=7251)
################################################
from sklearn.preprocessing import MinMaxScaler , StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
scaler = RobustScaler()

scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)
################################################

#2. 모델
xgb = XGBClassifier()
rf = RandomForestClassifier()
cat = CatBoostClassifier()

model = XGBClassifier()
model = VotingClassifier(
    estimators= [('xgb', xgb), ('rf', rf), ('cat', cat)],
    voting='soft',
    # voting='hard', # 디폴트
)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가 예측
results = model.score(x_test, y_test)
print('예측 점수 :', results )

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('acc 점수 :', acc)

# XGB


