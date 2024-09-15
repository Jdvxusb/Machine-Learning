import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv('maaslar.csv')

x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]
X = x.values
Y = y.values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)
lin_reg.fit(x_poly,y)

plt.scatter(X,Y)
plt.plot(x,lin_reg.predict(poly_reg.fit_transform(X)))
plt.show()