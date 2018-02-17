# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 15:33:46 2017

@author: Karthikeyan Sankar
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#from sklearn.linear_model import LinearRegression
#reg1 = LinearRegression()
#reg1.fit(X,y)
#is160 = reg1.predict(6.5)
#
#plt.scatter(X,y, color='red')
#plt.plot(X, reg1.predict(X), color='blue')
#plt.xlabel('POS')
#plt.ylabel('Sal')
#plt.show()
#
#from sklearn.preprocessing import PolynomialFeatures
#polyreg = PolynomialFeatures(degree = 5)
#X_poly = polyreg.fit_transform(X)
#polyreg.fit(X_poly,y)
#reg2 = LinearRegression()
#reg2.fit(X_poly,y)
#
#plt.scatter(X,y, color='red')
#plt.plot(X, reg2.predict(polyreg.fit_transform(X)), color='blue')
#plt.xlabel('POS')
#plt.ylabel('Sal')
#plt.show()

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X = sc_x.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1,1))
y = np.ravel(y)

from sklearn.svm import SVR
reg3 = SVR(kernel = 'rbf')
reg3.fit(X,y)

y_pred = reg3.predict(sc_x.transform(np.array([[6.5]])))
#X = sc_x.inverse_transform(X)
#y = sc_y.inverse_transform(y)
y_pred = sc_y.inverse_transform(y_pred)

plt.scatter(X,y, color='red')
plt.plot(X, reg3.predict(X), color='blue')
plt.xlabel('POS')
plt.ylabel('Sal')
plt.show()
