from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.datasets import load_boston
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

tf.random.set_seed(337)
np.random.seed(337)

datasets = load_boston()
df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
print(df)
df['target'] = datasets.target

# df.boxplot() # TAX
# df.plot.box()
# plt.show()

x = df.drop(['target'], axis=1).copy()
y = df['target']
##################### x population로그 변환 #######################
x['TAX'] = np.log1p(x['TAX']) # 지수변환 np.exp1m

# train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=1234,
)

############################### y 로그 변환 ########################
y_train = np.log1p(y_train)
y_test = np.log1p(y_test)

# 2 모델
model = RandomForestRegressor(random_state=1234,
                              max_depth=5,
                              min_samples_split=3,)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가 예측
score = model.score(x_test, y_test)
print('score :', score)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2 score :', r2)

# 로그 변환전
# score : 0.8905816885452467
# r2 score : 0.8905816885452467

# 로그 변환 x만 한 후
# score : 0.8906206590324358
# r2 score : 0.8906206590324358

# 로그 변환  x,y둘 다
# score : 0.8575446276395677
# r2 score : 0.8575446276395677





























































