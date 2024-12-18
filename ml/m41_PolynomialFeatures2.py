import numpy as np
from sklearn.preprocessing import PolynomialFeatures

x = np.arange(12).reshape(4, 3)
print(x)





pf = PolynomialFeatures(degree=2, include_bias=False)
x_pf = pf.fit_transform(x)
print(x_pf)

pf = PolynomialFeatures(degree=2, include_bias=True)
x_pf = pf.fit_transform(x)
print(x_pf)



pf = PolynomialFeatures(degree=3, include_bias=False)
x_pf = pf.fit_transform(x)
print(x_pf)




