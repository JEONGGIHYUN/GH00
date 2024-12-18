import numpy as np
import pandas as pd
from sklearn.datasets import load_linnerud
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ridge_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import StackingClassifier, BaggingClassifier, VotingClassifier
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

x, y = load_linnerud(return_X_y=True)
print(x.shape, y.shape) # (20, 3) (20, 3)

print(x)

print(y)

#2. 모델
model = RandomForestRegressor()
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 :',
      round(mean_absolute_error(y, y_pred), 4))
print(model.predict([[2, 110, 4]]))
# RandomForestRegressor 스코어 : 3.6113
# [[159.62  34.8   61.82]]

#2. 모델
model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 :',
      round(mean_absolute_error(y, y_pred), 4))
print(model.predict([[2, 110, 4]]))

# LinearRegression 스코어 : 7.4567
# [[183.7070079   35.99900071  56.55115281]]

# #2. 모델
# model = Ridge()
# model.fit(x, y)
# y_pred = model.predict(x)
# print(model.__class__.__name__, '스코어 :',
#       round(mean_absolute_error(y, y_pred), 4))
# print(model.predict([[2, 110, 4]]))


#2. 모델
model = XGBRegressor()
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 :',
      round(mean_absolute_error(y, y_pred), 4))
print(model.predict([[2, 110, 4]]))

# XGBRegressor 스코어 : 0.0008
# [[138.54855   33.2387    60.323563]]

from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier

#2. 모델
model = CatBoostRegressor(loss_function='MultiRMSE')
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 :',
    round(mean_absolute_error(y, y_pred), 4))
print(model.predict([[2, 110, 4]]))

# MultiOutputRegressor(CatBoostRegressor) 스코어 : 0.2154
# [[152.45245094  33.72368268  60.79052202]]

# CatBoostRegressor 스코어 : 0.0638
# [[152.73895698  34.16151893  62.81759669]]

#2. 모델
# model = MultiOutputRegressor(LGBMRegressor())
# model.fit(x, y)
# y_pred = model.predict(x)
# print(model.__class__.__name__,'스코어 :',
    #   round(mean_absolute_error(y, y_pred), 4))
# print(model.predict([[2, 110, 4]]))
# MultiOutputRegressor 스코어 : 8.91
# [[178.6  35.4  56.1]]

def create_Multi_class_data_with_labels():
    x = np.random.rand(20, 3)

    y = np.random.randint(0, 10, size=[20, 3]) # 각 클래스에 0 부터 9까지 값

# 데이터 프레임으로 변환
    x_df = pd.DataFrame(x, columns=['Feature1','Feature2', 'Feature3'])
    y_df = pd.DataFrame(y, columns=['Label1','Label2', 'Label3'])

    return x_df, y_df

x, y = create_Multi_class_data_with_labels()

print(x)

print(y)




































































