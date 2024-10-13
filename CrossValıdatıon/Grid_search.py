import pandas as pd

veriler = pd.read_excel('Iris.xls')

x = veriler.iloc[:,0:4].values
y = veriler.iloc[:,4].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.25, random_state=0)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state=0)
classifier.fit(x_train,y_train)

from sklearn.model_selection import GridSearchCV
p = [{'C':[1,2,3,4,5],'kernel':['linear','rbf']},
     {'C':[1,10,100.1000],'kernel':['rbf'],'gamma':[1,0.5,0.1,0.01,0.001]}]

gs = GridSearchCV(estimator=classifier,param_grid=p,scoring='accuracy',cv=10,n_jobs=-1)

grid_search = gs.fit(x_train, y_train)
eniyisonuc = grid_search.best_score_
eniyiparametreler = grid_search.best_params_

print(eniyisonuc)
print(eniyiparametreler)