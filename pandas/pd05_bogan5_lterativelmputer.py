import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# import xgb import 

data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                     [2, 4, np.nan, 8, np.nan],
                     [2, 4, 6, 8, 10],
                     [np.nan, 4, np.nan, 8, np.nan]])

print(data)
data = data.transpose()
data.columns = ['x1', 'x2', 'x3', 'x4']
print(data)

from sklearn.impute import IterativeImputer


imputer = IterativeImputer() # 디폴트 bayesianRidge 회귀모델. 
data1 = imputer.fit_transform(data) 
print(data1)

imputer = IterativeImputer(estimator=DecisionTreeRegressor()) # 디폴트 bayesianRidge 회귀모델. 
data2 = imputer.fit_transform(data) 
print(data2)

imputer = IterativeImputer(estimator=RandomForestRegressor()) # 디폴트 bayesianRidge 회귀모델. 
data3 = imputer.fit_transform(data) 
print(data3)

imputer = IterativeImputer(estimator=RandomForestRegressor()) # 디폴트 bayesianRidge 회귀모델. 
data4 = imputer.fit_transform(data) 
print(data4)







































