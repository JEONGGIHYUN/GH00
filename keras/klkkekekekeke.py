import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import numpy as np


dataset = fetch_california_housing()

x = dataset.data

y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=123465, shuffle=True)

xx = pd.read_csv('C:/TDS/ai5/_data/dacon/diabetes/test.csv')

print(xx)

# xx = xx.()

print(xx) # 변수.head, 변수.info 등등 은 변수 안에 데이터가 csv나 excel등의 형태일 때만 사용 가능하다.

