# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 15:56:16 2017

@author: Karthikeyan Sankar
"""

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding dummy variable trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fitting mlr to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting the test results
y_pred = regressor.predict(X_test)

#building a optimal model using backward eliminiation
import statsmodels.formula.api as sm
X = np.append(arr=np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_ols = sm.OLS(endog= y, exog= X_opt).fit()
regressor_ols.summary()
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_ols = sm.OLS(endog= y, exog= X_opt).fit()
regressor_ols.summary()
X_opt = X[:, [0, 3, 4, 5]]
regressor_ols = sm.OLS(endog= y, exog= X_opt).fit()
regressor_ols.summary()
X_opt = X[:, [0, 3, 5]]
regressor_ols = sm.OLS(endog= y, exog= X_opt).fit()
regressor_ols.summary()
X_opt = X[:, [0, 3]]
regressor_ols = sm.OLS(endog= y, exog= X_opt).fit()
regressor_ols.summary()

#Model with opt solution
from sklearn.cross_validation import train_test_split
X_opts = X[:, 3:4]
X_optstrain, X_optstest, y_optstrain, y_optstest = train_test_split(X_opts, y, test_size = 0.2, random_state = 0)

#Fitting mlr to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_optstrain, y_optstrain)

#predicting the test results
y_optspred = regressor.predict(X_optstest)

