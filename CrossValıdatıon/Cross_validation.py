import pandas as pd

veriler = pd.read_excel('Iris.xls')

x = veriler.iloc[:,0:4].values
y = veriler.iloc[:,4].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.25, random_state=0)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state=0)
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)

from sklearn.model_selection import cross_val_score
basari = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv = 4)
print(basari.mean())
print(basari.std())