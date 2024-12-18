# from sklearn.datasets import load_iris
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from xgboost import XGBClassifier
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

# #1. 데이터
# # x, y = load_iris(return_X_y=True)
# # print(x.shape, y.shape)
# # (150, 4) (150,)

# # datasets = load_iris()
# # x = datasets.data
# # y = datasets.target


# # random_state = 122
# # from sklearn.model_selection import train_test_split
# # x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=random_state, stratify=y)

# # #2. 모델 구성

# # model = XGBClassifier(random_state=random_state)



# # i=0
# # print('random_state :', random_state)
# # model.fit(x_train, y_train)
# # print('===========', model.__class__.__name__, '==============')
# # print('acc', model.score(x_test, y_test))
# # print(model.feature_importances_)

# # def plot_feature_importances_dataset(model):
# #     n_features =datasets.data.shape[1]
# #     plt.barh(np.arange(n_features), model.feature_importances_,
# #          align='center')
# #     plt.yticks(np.arange(n_features), datasets.feature_names)
# #     plt.xlabel('Feature Importances')
# #     plt.ylabel('Features')
# #     plt.ylim(-1, n_features)
# #     plt.title(model.__class__.__name__)
# # plt.subplot(2, 2, i+1)
# # plot_feature_importances_dataset(model)
# # i=i+1
# # plt.tight_layout()
# # plt.show()

# datasets = load_iris()
# x = pd.DataFrame(datasets.data)
# y = pd.DataFrame(datasets.target)

# print(x)

# # x = x.drop(['2', '3'])

# # print(x['feature_names']) # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']



# random_state = 122
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=random_state, stratify=y)

# #2. 모델 구성

# model = XGBClassifier(random_state=random_state)



# i=0
# print('random_state :', random_state)
# model.fit(x_train, y_train)
# print('===========', model.__class__.__name__, '==============')
# print('acc', model.score(x_test, y_test))
# print(model.feature_importances_)

# def plot_feature_importances_dataset(model):
#     n_features =datasets.data.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_,
#          align='center')
#     plt.yticks(np.arange(n_features), datasets.feature_names)
#     plt.xlabel('Feature Importances')
#     plt.ylabel('Features')
#     plt.ylim(-1, n_features)
#     plt.title(model.__class__.__name__)
# plt.subplot(2, 2, i+1)
# plot_feature_importances_dataset(model)
# i=i+1
# plt.tight_layout()
# plt.show()


###################################################################################

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd

# 1. 데이터
datasets = load_iris()
df = pd.DataFrame(data=datasets.data, columns=datasets.feature_names)
x = df.values
y = datasets.target

# 데이터 분할
random_state1 = 1223
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=random_state1)

# 2. 모델 구성
model1 = DecisionTreeRegressor(random_state=random_state1)
model2 = RandomForestRegressor(random_state=random_state1)
model3 = GradientBoostingRegressor(random_state=random_state1)
model4 = XGBRegressor(random_state=random_state1)

models = [model1, model2, model3, model4]

for model in models:
    model.fit(x_train, y_train)
    feature_importances = model.feature_importances_
    
    # 중요도 기반 정렬 (내림차순)
    sorted_idx = np.argsort(feature_importances)
    
    print(f"\n================= {model.__class__.__name__} =================")
    print('Original R2 Score:', r2_score(y_test, model.predict(x_test)))
    print('Original Feature Importances:', feature_importances)
    
    # 하위 10%, 20%, 30%, 40%, 50% 제거하고 R2 스코어 계산
    for percentage in [10, 20, 30, 40, 50]:
        n_remove = int(len(sorted_idx) * (percentage / 100))
        removed_features_idx = sorted_idx[:n_remove]  # 하위 n% 특성 제거
        
        # 제거된 특성을 제외한 데이터셋 생성
        x_train_reduced = np.delete(x_train, removed_features_idx, axis=1)
        x_test_reduced = np.delete(x_test, removed_features_idx, axis=1)
        
        # 모델 재학습 및 평가
        model.fit(x_train_reduced, y_train)
        r2_reduced = r2_score(y_test, model.predict(x_test_reduced))
        
        print(f"R2 Score after removing {percentage}% lowest importance features: {r2_reduced}")
\
####################################################################################################

from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

import numpy as np

random_state = 7777

x, y = load_diabetes(return_X_y = True)

print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.8,
    random_state = random_state
)

model = RandomForestRegressor(random_state = random_state)

model.fit(x_train, y_train)

pencentile = np.percentile(model.feature_importances_, 15)

rm_index = []

for index, importance in enumerate(model.feature_importances_):
    if importance <= pencentile:
        rm_index.append(index)

x_train = np.delete(x_train, rm_index, axis = 1)
x_test = np.delete(x_test, rm_index, axis = 1)

model = RandomForestRegressor(random_state = random_state)

model.fit(x_train, y_train)

print("-------------------", model.__class__.__name__, "-------------------")
print('acc :', model.score(x_test, y_test))
print(model.feature_importances_)









































