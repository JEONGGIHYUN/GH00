import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
plt.rcParams['font.family'] = 'Malgun Gothic'

#1 데이터
np.random.seed(888)
x = 2 * np.random.rand(100, 1) -1 # -1부터 1까지 난수생성
# print(x)

print(np.max(x), np.min(x))

y = 3 * x**2 + 2 * x + np.random.randn(100, 1) # y = 3x^2 + 2x + 1 + 노이즈

# rand = 균등 분포 randn = 정규 분포

# rand = 0 ~ 1 사이의 균일분포 난수
# randn = 평균 0, 표편1 정규분포 난수

pf = PolynomialFeatures(degree=2, include_bias=True)

x_pf = pf.fit_transform(x)

#2. 모델
model = LinearRegression()
model2 = LinearRegression()

#3. 훈련
model.fit(x, y)
model2.fit(x_pf, y)

# 원래 데이터 그리기
plt.scatter(x, y, color='blue', label='original data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polinomial Features 예제')
# plt.show()

# 다항식 회귀 그래프 그리기
x_test = np.linspace(-1, 1, 100).reshape(-1, 1)
x_test_poly = pf.transform(x_test)
y_plot = model.predict(x_test)
y_plot2 = model.predict(x_test_poly)
plt.plot(x_test, y_plot, color='red', label='None')
plt.plot(x_test, y_plot2, color='green', label='Polynomial Regression')

plt.legend()
plt.show()

















