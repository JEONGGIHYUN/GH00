from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import numpy as np

#1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets.target
print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=7,train_size=0.8,shuffle=True,stratify=y) 

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

pca = PCA(n_components=3)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)


print(x)
print(x.shape) # (150, 3)



#2. 모델
model = RandomForestClassifier(random_state=4)

#3. 훈련
model.fit(x_train,y_train)

#4. 평가 예측
results = model.score(x_test,y_test)
print(x.shape)
print('model.score : ', results)

evr = pca.explained_variance_ratio_ # 설명가능한 변화율
print(evr)
print(sum(evr))

evr_cumsum = np.cumsum(evr)
print(evr_cumsum)

import matplotlib.pyplot as plt
plt.plot(evr_cumsum)
plt.grid()
plt.show()