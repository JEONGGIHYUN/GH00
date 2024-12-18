from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

#1. 데이터
datasets = fetch_california_housing()
df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
print(df)
df['target'] = datasets.target

# df.boxplot() # Population 이상치 발생
# df.plot.box()
# plt.show()

# print(df.info())
# print(df.describe())

# df['Population'].boxplot() 이건 시리즈 사용 불가능
# df['Population'].plot.box()
# plt.show()

# df['Population'].hist(bins=50)
# plt.show()

# df['target'].hist(bins=50)
# plt.show()

x = df.drop(['target'], axis=1).copy()
y = df['target']
##################### x population로그 변환 #######################
x['Population'] = np.log1p(x['Population']) # 지수변환 np.exp1m

# train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=1234,
)

############################### y 로그 변환 ########################
y_train = np.log1p(y_train)
y_test = np.log1p(y_test)


#2 모델
# model = RandomForestRegressor(random_state=1234,
                            #   max_depth=5,
                            #   min_samples_split=3,)

#2 모델 리니어 리그레서
model = LinearRegression()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가 예측
score = model.score(x_test, y_test)
print('score :', score)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2 score :', r2)

# 로그 변환전
# score : 0.6495152533878351
# r2 score : 0.6495152533878351

# 로그 변환 x만 한 후
# score : 0.6495031475648194
# r2 score : 0.6495031475648194

# 로그 변환  x,y둘 다
# score : 0.6584197269397019
# r2 score : 0.6584197269397019

# 로그 변환 전 리니어
# score : 0.606572212210644
# r2 score : 0.606572212210644

# 로그 변환  x만 한 후
# score : 0.606598836886877
# r2 score : 0.606598836886877

# 로그 변환 x,y 한 후
# score : 0.6294707351612604
# r2 score : 0.6294707351612604

