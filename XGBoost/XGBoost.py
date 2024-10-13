import pandas as pd

veriler = pd.read_csv('Churn_Modelling.csv')

x = veriler.iloc[:,3:13].values
y = veriler.iloc[:,13]

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
x[:,1] = le.fit_transform(x[:,1])
x[:,2] = le.fit_transform(x[:,2])

from sklearn.compose import ColumnTransformer
ohe = ColumnTransformer([('ohe', OneHotEncoder(dtype=float),[1])], remainder = 'passthrough')
x = ohe.fit_transform(x)
x = x[:,1:]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.33,random_state=0)

from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)