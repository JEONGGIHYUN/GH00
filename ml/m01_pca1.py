from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

#1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets.target
print(x.shape, y.shape)

# scaler = StandardScaler()
# x = scaler.fit_transform(x)

pca = PCA(n_components=4)
x = pca.fit_transform(x)


print(x)
print(x.shape) # (150, 3)

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=4,train_size=0.8,shuffle=True,stratify=y) 

#2. 모델
model = RandomForestClassifier(random_state=4)

#3. 훈련
model.fit(x_train,y_train)

#4. 평가 예측
results = model.score(x_test,y_test)
print(x.shape)
print('model.score : ', results)


# model.score :  0.9333333333333333 (150, 4) pca안한거

# model.score :  0.9 (150, 3) pca만 한것
 
# model.score :  0.9333333333333333 (150, 4) 둘 다 안한것

# model.score :  0.9333333333333333 (150, 2)




















































































