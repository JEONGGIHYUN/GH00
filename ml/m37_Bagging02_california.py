import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression #(로지스틱 리그레서는 회귀가 아닌 분류이다.)

#1. 데이터
x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1367)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
# model = DecisionTreeRegressor()
# model = BaggingRegressor(DecisionTreeRegressor(),
#                           n_estimators=100,
#                           n_jobs=-1,
#                           random_state=4444,
#                           bootstrap=True, # 디폴트True 중복을 허용한다.
#                           )
model = RandomForestRegressor()


#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수 : ', results)
y_pre = model.predict(x_test)
results = r2_score(y_test, y_pre)
print('results_score : ', results)

# 디시전트리리그레서 사용
# 최종점수 :  0.5871071972213249
# acc_score :  0.5871071972213249

# 디시전 배깅 사용
# 최종점수 :  0.7997136181571808
# results_score :  0.7997136181571808

# 랜덤 포레스트리그레서 사용
# 최종점수 :  0.8009744206496339
# results_score :  0.8009744206496339
