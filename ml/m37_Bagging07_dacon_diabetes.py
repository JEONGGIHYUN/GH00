import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score
from bayes_opt import BayesianOptimization
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression #(로지스틱 리그레서는 회귀가 아닌 분류이다.)
import time
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

#1. 데이터
path = './_data/dacon/diabetes/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
# print(train_csv) # [652 rows x 9 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=0)
# print(test_csv) # [116 rows x 8 columns]

submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)
# print(submission_csv)

# print(train_csv.shape) # (652, 9)
# print(test_csv.shape) # (116, 8)
# print(submission_csv.shape) # (116, 1)

# print(train_csv.columns)
# Index(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
    #    'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'],
    #   dtype='object')

x = train_csv.drop(['Outcome'], axis=1)

y = train_csv['Outcome']

random_state=888
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state= random_state, stratify=y)

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
                        #   )
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
# 최종점수 :  0.648854961832061
# acc_score :  0.648854961832061

# 디시전 배깅 사용
# 최종점수 :  0.6717557251908397
# acc_score :  0.6717557251908397

#로지스터 리그레션 사용
# 최종점수 :  0.7404580152671756
# acc_score :  0.7404580152671756

# 로지스틱 리그레션 배깅 사용
# 최종점수 :  0.7404580152671756
# acc_score :  0.7404580152671756

# 로지스틱 리그레션 배깅 사용 부트스트랩 펄스
# 최종점수 :  0.7404580152671756
# acc_score :  0.7404580152671756

# 랜덤포레스트 사용
# 최종점수 :  0.7404580152671756
# acc_score :  0.7404580152671756
