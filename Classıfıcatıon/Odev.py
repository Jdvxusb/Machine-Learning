import pandas as pd
import numpy as np

veriler = pd.read_excel('iris.xls')
veriler = veriler.drop(['sepal width'],axis=1)

x = veriler.iloc[:,0:3].values

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
y = le.fit_transform(veriler.iloc[:,3:])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=12)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

import statsmodels.api as sm
x = np.append(arr = np.ones((150,1)).astype(int), values=x, axis=1)

X_l = veriler.iloc[:,[0,1,2]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(y,X_l).fit()

print(model.summary())

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='entropy')

dtc.fit(x_train, y_train)
y_pred = dtc.predict(x_test)

from sklearn.metrics import confusion_matrix
print('DTC')
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.svm import SVC
svc = SVC(kernel = 'rbf')
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)

print('SVM')
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=12, criterion='entropy')
rfc.fit(x_train, y_train)
y_pred = rfc.predict(x_test)                        
     
print('RFC')
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=5)
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

cm = confusion_matrix(y_pred, y_test)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)
print('logistic regression')
cm = confusion_matrix(y_test, y_pred)
print(cm)