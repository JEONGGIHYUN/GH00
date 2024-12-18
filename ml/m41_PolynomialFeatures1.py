# x1 x2 칼럼이 있다면 
# 단항식이라면 다항식으로 제곱을 만들어보자 
# 데이터 증폭 (다항식으로 증폭)

import numpy as np
from sklearn.preprocessing import PolynomialFeatures

x = np.arange(8).reshape(4, 2)

print(x) 
# [0, 4]
# [1, 5]
# [2, 6]
# [3, 7]

# pf = PolynomialFeatures(degree=2, include_bias=False) # 디폴트 = True
# x_pf = pf.fit_transform(x)
# print(x_pf)

# pf = PolynomialFeatures(degree=2, include_bias=True) # 디폴트 = True
# x_pf = pf.fit_transform(x)
# print(x_pf)

# ### 통상적으로
# 선형모델(lr등)에 쓸 경우에는 include_bias = True를 써서 1만 있는 컬럼을 만드는게 좋고,
# 왜냐하면 y = wx+b의 바이어스=1의 역할을 하기 때문
# 비선형모델(rf, xgb 등)에 쓸 경우에는 include_bias = False가 좋다.

pf = PolynomialFeatures(degree=3, include_bias=False)
x_pf = pf.fit_transform(x)
print(x_pf)




































































