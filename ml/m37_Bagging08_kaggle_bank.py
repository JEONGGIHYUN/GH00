import numpy as np
from sklearn.datasets import load_breast_cancer
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
path = './_data/kaggle/playground-series-s4e1/'

train_csv = pd.read_csv(path + 'train.csv', index_col=[0,1,2])
# print(train_csv) # [165034 rows x 13 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=[0,1,2]) 
# print(test_csv) #  [110023 rows x 12 columns]

submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0) 
# print(submission_csv) # [110023 rows x 1 columns]

# print(train_csv.shape) # (165034, 13)
# print(test_csv.shape) # (110023, 12)
# print(submission_csv.shape) # (110023, 1)

# print(train_csv.columns)
# Index(['CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender', 'Age',
#        'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
#        'EstimatedSalary', 'Exited'],
#       dtype='object')
'''
df = pd.DataFrame(train_csv)

# train_csv = train_csv['Geography'].str.replace('France', '1')
df = df.replace({'Geography':'France'}, '0')
df = df.replace({'Geography':'Germany'}, '1')
df = df.replace({'Geography':'Spain'}, '2')

df = df.replace({'derGen':'Male'}, '1')
df = df.replace({'derGen':'Female'}, '0')
'''
geography_mapping = {'France': 0, 'Germany': 1, 'Spain': 2}
derGen_mapping = {'Male': 1, 'Female': 0}

train_csv['Geography'] = train_csv['Geography'].map(geography_mapping)
train_csv['Gender'] = train_csv['Gender'].map(derGen_mapping)

test_csv['Geography'] = test_csv['Geography'].map(geography_mapping)
test_csv['Gender'] = test_csv['Gender'].map(derGen_mapping)


x = train_csv.drop(['Exited'], axis=1)
y = train_csv['Exited']


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
# 최종점수 :  0.7923167812888176
# acc_score :  0.7923167812888176

# 디시전 배깅 사용
# 최종점수 :  0.8543036325627897
# acc_score :  0.8543036325627897

#로지스터 리그레션 사용
# 최종점수 :  0.828248553337171
# acc_score :  0.828248553337171

# 로지스틱 리그레션 배깅 사용
# 최종점수 :  0.828127366922168
# acc_score :  0.828127366922168

# 로지스틱 리그레션 배깅 사용 부트스트랩 펄스
# 최종점수 :  0.828248553337171
# acc_score :  0.828248553337171

# 랜덤포레스트 사용
# 최종점수 :  0.8576968521828703
# acc_score :  0.8576968521828703




