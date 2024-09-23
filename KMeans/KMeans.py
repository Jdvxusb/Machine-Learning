import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv('musteriler.csv')

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
veriler['Cinsiyet'] = le.fit_transform(veriler['Cinsiyet'])

x = veriler.iloc[:,1:4].values
y = veriler.iloc[:,4:].values

from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, init = 'k-means++')

sonuclar = []
for i in range(1,11):
    km = KMeans(n_clusters=i, init='k-means++', random_state=5)
    km.fit(x)
    sonuclar.append(km.inertia_)
    
plt.plot(range(1,11),sonuclar)
plt.show()