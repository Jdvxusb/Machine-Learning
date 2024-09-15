import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv('maaslar.csv')

x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]
X = x.values
Y = y.values

from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_olcek = sc1.fit_transform(X)
y_olcek = sc1.fit_transform(Y)

from sklearn.svm import SVR

svr_reg = SVR(kernel = 'rbf')
svr_reg.fit(x_olcek,y_olcek)

plt.scatter(x_olcek,y_olcek)
plt.plot(x_olcek,svr_reg.predict(x_olcek))
plt.show()