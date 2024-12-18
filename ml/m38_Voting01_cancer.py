import numpy as np
from sklearn.datasets import load_breast_cancer
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
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1367, stratify=y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
xgb = XGBClassifier()
rf = RandomForestClassifier()
cat = CatBoostClassifier()

# model = XGBClassifier()
model = VotingClassifier(
    estimators= [('xgb', xgb), ('rf', rf), ('cat', cat)],
    # voting='soft',
    voting='hard', # 디폴트
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
# 예측 점수 : 0.9649122807017544
# acc 점수 : 0.9649122807017544

# soft
# 예측 점수 : 0.9736842105263158
# acc 점수 : 0.9736842105263158

# hard
# 예측 점수 : 0.9824561403508771
# acc 점수 : 0.9824561403508771












