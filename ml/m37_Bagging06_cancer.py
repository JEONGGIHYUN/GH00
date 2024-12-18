import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression #(로지스틱 리그레서는 회귀가 아닌 분류이다.)

#1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1367, stratify=y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
# model = DecisionTreeClassifier()
# model = BaggingClassifier(DecisionTreeClassifier(),
#                           n_estimators=100,
#                           n_jobs=-1,
#                           random_state=4444,
#                           bootstrap=True, # 디폴트True 중복을 허용한다.
#                           )
# model = LogisticRegression()
# model = BaggingClassifier(LogisticRegression(),
#                           n_estimators=100,
#                           n_jobs=-1,
#                           random_state=4444,
#                           bootstrap=True, # 디폴트True 중복을 허용한다
#                           )
model = RandomForestClassifier()


#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수 : ', results)
y_pre = model.predict(x_test)
acc = accuracy_score(y_test, y_pre)

print('acc_score : ', acc)

# 디시전
# 최종점수 :  0.9473684210526315
# acc_score :  0.9473684210526315

# 디시전 배깅 사용
# 최종점수 :  0.9824561403508771
# acc_score :  0.9824561403508771

#로지스터 리그레션 사용
# 최종점수 :  0.9736842105263158
# acc_score :  0.9736842105263158

# 로지스틱 리그레션 배깅 사용
# 최종점수 :  0.9736842105263158
# acc_score :  0.9736842105263158

# 로지스틱 리그레션 배깅 사용 부트스트랩 펄스
# 최종점수 :  0.9736842105263158
# acc_score :  0.9736842105263158

# 랜덤포레스트 사용
# 최종점수 :  0.9736842105263158
# acc_score :  0.9736842105263158



























