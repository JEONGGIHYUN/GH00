
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier

# 함수 호출 및 데이터 출력

def create_multiclass_data_with_labels():
    np.random.seed(111)
    x = np.random.rand(20, 3)
    y = np.random.randint(0, 5, size=(20, 3))
    
    X_df = pd.DataFrame(x, columns=['Feature1', 'Feature2', 'Feature3'])
    y_df = pd.DataFrame(y, columns=['Label1', 'Lebel2', 'Label3'])
    
    return X_df, y_df

x, y = create_multiclass_data_with_labels()

# #2. 모델
# model = RandomForestClassifier()
# model.fit(x, y)
# y_pred = model.predict(x)
# print(model.__class__.__name__, '스코어 :',
#       round(mean_absolute_error(y, y_pred), 4))
# print(model.predict([[0.195983, 0.045227, 0.325330]]))

# RandomForestClassifier 스코어 : 0.0
# [[2 4 3]]



# #2. 모델
# model = LinearRegression()
# model.fit(x, y)
# y_pred = model.predict(x)
# print(model.__class__.__name__, '스코어 :',
    #   round(mean_absolute_error(y, y_pred), 4))
# print(model.predict([[0.195983, 0.045227, 0.325330]]))

# LinearRegression 스코어 : 1.0497
# [[2.49493182 1.66713037 2.27780675]]



# #2. 모델
# model = Ridge()
# model.fit(x, y)
# y_pred = model.predict(x)
# print(model.__class__.__name__, '스코어 :',
    #   round(mean_absolute_error(y, y_pred), 4))
# print(model.predict([[0.195983, 0.045227, 0.325330]]))



# 2. 모델
# model = MultiOutputClassifier(XGBClassifier())
# model.fit(x, y)
# y_pred = model.predict(x)
# print(model.__class__.__name__, '스코어 :',
    #   round(mean_absolute_error(y, y_pred), 4))
# print(model.predict([[0.195983, 0.045227, 0.325330]]))

# MultiOutputClassifier 스코어 : 0.0
# [[2 2 4]]



#2. 모델
model = MultiOutputClassifier(CatBoostClassifier()) # CatBoostClassifier는 함수 지원을 안함
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 :',
    round(mean_absolute_error(y, y_pred.reshape(20, 3)), 4))
print(model.predict([[0.195983, 0.045227, 0.325330]]))

# MultiOutputClassifier 스코어 : 0.0
# [[[2 4 3]]]



#2. 모델
model = MultiOutputClassifier(LGBMClassifier())
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__,'스코어 :',
      round(mean_absolute_error(y, y_pred), 4))
print(model.predict([[0.195983, 0.045227, 0.325330]]))

# MultiOutputClassifier 스코어 : 1.45
# [[2 0 4]]

exit()
