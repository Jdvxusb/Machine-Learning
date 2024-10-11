import pandas as pd

veriler = pd.read_csv('Wine.csv')

x = veriler.iloc[:,0:13].values
y = veriler.iloc[:,13].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

from sklearn.decomposition import PCA 
pca = PCA(n_components=2)

x_train2 = pca.fit_transform(x_train)
x_test2 = pca.transform(x_test)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=0)
lr.fit(x_train2,y_train)

y_pred = lr.predict(x_test2)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("PCA")
print(cm)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
x_train_lda = lda.fit_transform(x_train,y_train)
x_test_lda = lda.transform(x_test)

lr2 = LogisticRegression(random_state=0)
lr2.fit(x_train_lda, y_train)

y_pred2 = lr2.predict(x_test_lda)
cm2 = confusion_matrix(y_test, y_pred2)
print('LDA')
print(cm2)