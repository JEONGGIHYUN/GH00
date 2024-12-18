# california, diabet
### 회귀 ###
# 랜포 그라디언트부스트 xg부스트 같은 트리형 구조는 피쳐 임포턴스를 제공한다.
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

#1. 데이터
data1 = fetch_california_housing(return_X_y=True)
data2 = load_diabetes(return_X_y=True)

loads = [data1,data2]
# (150, 4) (150,)

random_state = 1223
from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=random_state)

#2. 모델 구성

model1 = DecisionTreeRegressor(random_state=random_state)
model2 = RandomForestRegressor(random_state=random_state)
model3 = GradientBoostingRegressor(random_state=random_state)
model4 = XGBRegressor(random_state=random_state,)

models = [model1, model2, model3, model4]

print('random_state :', random_state)
for i in loads:
    x, y = i
    random_state = 1223
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=random_state)
    for model in models:
        model.fit(x_train, y_train)
        print('===========', model.__class__.__name__, '==============')
        print('acc', model.score(x_test, y_test))
        print(model.feature_importances_)
        
# random_state : 1223
# =========== DecisionTreeRegressor ==============
# acc 0.5964140465722068
# [0.51873533 0.05014494 0.05060456 0.02551158 0.02781676 0.13387334
#  0.09833673 0.09497676]
# =========== RandomForestRegressor ==============
# acc 0.811439104037621
# [0.52445075 0.05007899 0.04596161 0.03031591 0.03121773 0.1362301
#  0.09138102 0.09036389]
# =========== GradientBoostingRegressor ==============
# acc 0.7865333436969877
# [0.60051609 0.02978481 0.02084099 0.00454408 0.0027597  0.12535772
#  0.08997582 0.12622079]
# =========== XGBRegressor ==============
# acc 0.8384930657222394
# [0.49375907 0.06520814 0.04559402 0.02538511 0.02146595 0.14413244
#  0.0975963  0.10685894]
# =========== DecisionTreeRegressor ==============
# acc -0.24733855513252667
# [0.05676749 0.01855931 0.23978058 0.08279462 0.05873671 0.0639961 
#  0.04130515 0.01340568 0.33217096 0.0924834 ]
# =========== RandomForestRegressor ==============
# acc 0.3687286985683689
# [0.05394197 0.00931513 0.25953258 0.1125408  0.04297661 0.05293764
#  0.06684433 0.02490964 0.29157054 0.08543076]
# =========== GradientBoostingRegressor ==============
# acc 0.3647974813076822
# [0.04509096 0.00780692 0.25858035 0.09953666 0.02605597 0.06202725
#  0.05303144 0.01840481 0.35346141 0.07600423]
# =========== XGBRegressor ==============
# acc 0.10076704957922011
# [0.04070464 0.0605858  0.16995801 0.06239288 0.06619858 0.06474677
#  0.05363544 0.03795785 0.35376146 0.09005855]