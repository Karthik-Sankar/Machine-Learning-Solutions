# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 20:05:00 2017

@author: Karthikeyan Sankar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, [3,4,5,6,7,8,9,10,11,13,14]].values
y = dataset.iloc[:, -1].values

#Taking Care of missing data(2,6,7) -> all numeric
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='median')
imputer = imputer.fit(X[:, 2:3])
X[:,2:3] = imputer.transform(X[:,2:3])

imputer2 = Imputer(missing_values='NaN', strategy='mean')
imputer2 = imputer2.fit(X[:, 6:7])
X[:,6:7] = imputer2.transform(X[:,6:7])

imputer3 = Imputer(missing_values='NaN', strategy='median')
imputer3 = imputer3.fit(X[:, 7:8])
X[:,7:8] = imputer3.transform(X[:,7:8])

#Categorical Data (0,3,5,10)
from sklearn.preprocessing import LabelEncoder
label_enc = LabelEncoder()
X[:, 0] = label_enc.fit_transform(X[:,0])
X[:, 3] = label_enc.fit_transform(X[:,3])
X[:, 5] = label_enc.fit_transform(X[:,5])
X[:, 10] = label_enc.fit_transform(X[:,10])

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(categorical_features=[0])
X = encoder.fit_transform(X).toarray()
X = X[:,1:]
encoder = OneHotEncoder(categorical_features=[6])
X = encoder.fit_transform(X).toarray()
X = X[:,1:]
encoder = OneHotEncoder(categorical_features=[11])
X = encoder.fit_transform(X).toarray()
X = X[:,1:]
encoder = OneHotEncoder(categorical_features=[16])
X = encoder.fit_transform(X).toarray()
X = X[:,1:]

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
reg.score(X_train,y_train)


from statsmodels.formula.api import OLS
X_train = np.append(arr=np.ones((7492, 1)).astype(int), values = X_train, axis = 1)
X_opt = X_train[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,19,20]]
ols_reg = OLS(endog=y_train,exog=X_opt).fit()
ols_reg.summary()
#
#X_opt = X_train[:,[0,1,2,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,24,25,26,27,29]]
#ols_reg = OLS(endog=y_train,exog=X_opt).fit()
#ols_reg.summary()
#
X_opt = X_opt[:,1:]

reg2 = LinearRegression()
reg2.fit(X_opt,y_train)
y_pred2 = reg.predict(X_test)
reg2.score(X_opt,y_train)