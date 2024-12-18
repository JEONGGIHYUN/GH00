from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

# 1. 데이터
datasets = fetch_california_housing()
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
    
    # 하위 30% 특성 제거
    n_remove = int(len(sorted_idx) * 0.3)
    removed_features_idx = sorted_idx[:n_remove]  # 하위 30% 특성 제거
    
    # 제거된 특성들
    x_train_removed = x_train[:, removed_features_idx]
    x_test_removed = x_test[:, removed_features_idx]
    
    # PCA를 사용해 제거된 특성들을 병합
    pca = PCA(n_components=1)  # 주성분 1개로 변환
    x_train_pca = pca.fit_transform(x_train_removed)
    x_test_pca = pca.transform(x_test_removed)
    
    # PCA로 변환된 특성을 원래 데이터셋에 추가
    x_train_augmented = np.hstack((np.delete(x_train, removed_features_idx, axis=1), x_train_pca))
    x_test_augmented = np.hstack((np.delete(x_test, removed_features_idx, axis=1), x_test_pca))
    
    # 모델 재학습 및 평가
    model.fit(x_train_augmented, y_train)
    r2_reduced = r2_score(y_test, model.predict(x_test_augmented))
    
    print(f"R2 Score after removing {n_remove} features and adding PCA component: {r2_reduced}")