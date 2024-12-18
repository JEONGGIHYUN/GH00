import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression # <- 분류 모델
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# 1. 데이터
x, y = load_breast_cancer(return_X_y=True)

random_state=777
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    random_state=4444,
                                                    shuffle=True, 
                                                    train_size=0.8,
                                                    stratify=y
                                                    )

print(x_train.shape, x_test.shape) # (455, 30) (114, 30)
print(y_train.shape, y_test.shape) # (455,) (114,)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델

xgb = XGBClassifier()
rf = RandomForestClassifier()
cat = CatBoostClassifier(verbose=0)

train_list  = []
test_list = []
models = [xgb, rf, cat]

model = StackingClassifier(
    estimators=[('xgb',xgb),('rf', rf),('cat', cat)],
    final_estimator= CatBoostClassifier(verbose=0),
    n_jobs = -1,
    cv=5
)

# 3. 훈련

model.fit(x_train, y_train)

# 4. 평가 예측

results = model.score(x_test, y_test)
print('예측 점수 :', results )

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('스태킹 결과 :', acc)
