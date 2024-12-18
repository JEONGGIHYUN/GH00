import numpy as np
import pandas as pd

data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                     [2, 4, np.nan, 8, np.nan],
                     [2, 4, 6, 8, 10],
                     [np.nan, 4, np.nan, 8, np.nan]])

print(data)
data = data.transpose()
data.columns = ['x1', 'x2', 'x3', 'x4']
print(data)

from sklearn.impute import SimpleImputer, KNNImputer


imputer = SimpleImputer()
data2 = imputer.fit_transform(data)
print(data2)

imputer = SimpleImputer(strategy='mean')
data3 = imputer.fit_transform(data)
print(data3)

imputer = SimpleImputer(strategy='median')
data4 = imputer.fit_transform(data)
print(data4)

imputer = SimpleImputer(strategy='mmost_freqee')
data5 = imputer.fit_transform(data)
print(data5)

imputer = SimpleImputer(strategy='constant',)
data6 = imputer.fit_transform(data)
print(data6)

imputer = KNNImputer() # ㅏ
data7 = imputer.fit_transform(data) 
print(data6)

