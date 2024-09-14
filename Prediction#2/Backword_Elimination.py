import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
veriler = pd.read_csv('veriler.csv')

# Extract the 'ulke' column for encoding
ulke = veriler.iloc[:,0:1].values

# Label encode the 'ulke' column
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
ulke[:,0] = le.fit_transform(veriler.iloc[:,0])

# One-hot encode the 'ulke' column
ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()

# Extract and encode the 'cinsiyet' column
c = veriler.iloc[:,-1:].values
c[:, -1] = le.fit_transform(veriler.iloc[:, -1])
c = ohe.fit_transform(c).toarray()

# Create DataFrame for the 'ulke' column
sonuc = pd.DataFrame(data=ulke, index=range(veriler.shape[0]), columns=['fr', 'tr', 'us'])

# Drop 'cinsiyet' from the original data
veriler = veriler.drop('cinsiyet', axis=1)

# Create DataFrame for 'cinsiyet' after encoding
sonuc2 = pd.DataFrame(data=c[:, -1], index=range(veriler.shape[0]), columns=['cinsiyet'])

# Concatenate 'veriler' with the encoded 'cinsiyet' and 'ulke' columns
veriler = pd.concat([veriler, sonuc2, sonuc], axis=1)

# Drop the original 'ulke' column
veriler = veriler.drop('ulke', axis=1)

# Separate features (X) and target (y)
X = veriler.drop('boy',axis=1) 
y = veriler['boy'] 

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# Train a Linear Regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predict on test data
y_pred = regressor.predict(x_test)

import statsmodels.api as sm
x = np.append(arr = np.ones((22,1)).astype(int), values=X, axis=1)

X_l = X.iloc[:,[0,2,3,4,5]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(y,X_l).fit()

print(model.summary())