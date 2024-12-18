from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import numpy as np
import pandas as pd

datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

df = pd.DataFrame(x, columns=datasets.feature_names)
df['Target'] = y
print(df)

print('=============== 상관계수 히트맵 ==================')
print(df.corr())

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
print(sns.__version__)
print(matplotlib.__version__)
# sns.set(font_scale=1.2)
sns.heatmap(data= df.corr(),
            square=True,
            annot=True,
            cbar=True
            )
plt.show()