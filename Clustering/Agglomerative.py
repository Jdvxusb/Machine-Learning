import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv('musteriler.csv')

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
veriler['Cinsiyet'] = le.fit_transform(veriler['Cinsiyet'])

x = veriler.iloc[:,2:4].values
y = veriler.iloc[:,4:].values

from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='ward')
y_guess = ac.fit_predict(x)
print(y_guess)

plt.scatter(x[y_guess==0,0],x[y_guess==0,1],s=100,c='red')
plt.scatter(x[y_guess==1,0],x[y_guess==1,1],s=100,c='green')    
plt.scatter(x[y_guess==2,0],x[y_guess==2,1],s=100,c='blue')
plt.scatter(x[y_guess==3,0],x[y_guess==3,1],s=100,c='yellow')

plt.show()

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x,method='ward'))
plt.show()