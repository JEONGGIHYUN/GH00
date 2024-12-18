parameters =[
    {'n_jobs' : [-1,], 'n_estimators' : [100, 500], 'max_depth' : [6, 10, 12],
     'min_samples_leaf' : [3, 10]},
    {'n_jobs' : [-1,], 'max_depth' : [6, 7, 10, 12],
     'min_samples_leaf' : [3, 5, 7, 10]},
    {'n_jobs' : [-1,], 'min_samples_leaf' : [3, 5, 7, 10],
     'min_samples_split' : [2, 3, 5, 10]},   
    {'n_jobs' : [-1,], 'min_samples_split' : [2, 3, 5, 10]},   
]

n_splits = 5 
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=333)

parameters =[
    {'n_jobs' : [-1,], 'n_estimators' : [100, 500], 'max_depth' : [6, 10, 12],
     'min_samples_leaf' : [3, 10]},
    {'n_jobs' : [-1,], 'max_depth' : [6, 7, 10, 12],
     'min_samples_leaf' : [3, 5, 7, 10]},
    {'n_jobs' : [-1,], 'min_samples_leaf' : [3, 5, 7, 10],
     'min_samples_split' : [2, 3, 5, 10]},   
    {'n_jobs' : [-1,], 'min_samples_split' : [2, 3, 5, 10]},   
]


#2. 모델
model = GridSearchCV(XGBregresser(), parameters, cv=kfold,
                     verbose=1,
                     refit=True,
                     n_jobs=-1,
                     )

#3. 훈련
start_time = time.time()
model.fit(x_train,y_train)
end_time = time.time()

#4. 예측
print('최적의 매개변수 :', model.best_estimator_)

print('최적의 파라미터',model.best_params_)

print('best_score : ', model.best_score_)

print('model.score :', model.score(x_test,y_test))

y_pred = model.predict(x_test)
print('accuracy_score :', accuracy_score(y_test,y_pred))

y_pred_best = model.best_estimator_.predict(x_test)
print('accuracy_score :', accuracy_score(y_test, y_pred_best))

print('걸린 시간 :', round(end_time - start_time, 2), '초')
