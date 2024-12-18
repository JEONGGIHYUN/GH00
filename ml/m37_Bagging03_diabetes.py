import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression #(로지스틱 리그레서는 회귀가 아닌 분류이다.)
from sklearn.metrics import accuracy_score
from xgboost import XGBRegressor, XGBRFClassifier

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
model = DecisionTreeClassifier()
model = BaggingClassifier(DecisionTreeClassifier(),
                          n_estimators=100,
                          n_jobs=-1,
                          random_state=4444,
                          bootstrap=True, # 디폴트True 중복을 허용한다.
                          )
# model = LogisticRegression()
# model = BaggingClassifier(LogisticRegression(),
#                           n_estimators=100,
#                           n_jobs=-1,
#                           random_state=4444,
#                           bootstrap=True, # 디폴트True 중복을 허용한다
#                           )
# model = RandomForestClassifier()


#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수 : ', results)
y_pre = model.predict(x_test)
acc = accuracy_score(y_test, y_pre)

print('acc_score : ', acc)

# 디시전
# 최종점수 :  0.011235955056179775
# acc_score :  0.011235955056179775

# 디시전 배깅 사용
# 최종점수 :  0.011235955056179775
# acc_score :  0.011235955056179775


#로지스터 리그레션 사용
# 최종점수 :  0.011235955056179775
# acc_score :  0.011235955056179775


# 로지스틱 리그레션 배깅 사용
# 최종점수 :  0.0
# acc_score :  0.0

# 로지스틱 리그레션 배깅 사용 부트스트랩 펄스
# 최종점수 :  0.0
# acc_score :  0.0

# 랜덤포레스트 사용
# 최종점수 :  0.011235955056179775
# acc_score :  0.011235955056179775
