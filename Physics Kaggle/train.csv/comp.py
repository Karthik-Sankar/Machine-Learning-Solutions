# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 14:31:06 2018

@author: Karthikeyan Sankar
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('train.csv')

X = dataset.iloc[:, 1:12].values
y1 = dataset.iloc[:, 12].values
y2 = dataset.iloc[:, 13].values

#finding optimal features for independent variable 1(formation_energy_ev_natom)
import statsmodels.formula.api as sm
X = np.append(arr=np.ones((2400, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5, 6, 8, 9, 11]]
regressor_ols = sm.OLS(endog= y1, exog= X_opt).fit()
regressor_ols.summary()

X_ft1 = X_opt[:, 1:]

from sklearn.model_selection import train_test_split
X1_tr,X1_test,y1_tr,y1_test = train_test_split(X_ft1,y1,test_size=0.2,random_state=0)

#from sklearn.preprocessing import StandardScaler
#sc_x = StandardScaler()
#X1_tr = sc_x.fit_transform(X1_tr)
#X1_test = sc_x.transform(X1_test)

from sklearn.ensemble import RandomForestRegressor
reg1 = RandomForestRegressor(n_estimators= 20, criterion="mse", random_state=0)
reg1.fit(X1_tr,y1_tr)

y_pr = reg1.predict(X1_test)

#importing test data set

testdata = pd.read_csv('test.csv')
TX = testdata.iloc[:, [0,1,2,3,4,5,7,8,10]].values

ayans1 = reg1.predict(TX)


#finding optimal features for independent variable 2(bandgap_energy_ev)
import statsmodels.formula.api as sm
X2 = dataset.iloc[:, 1:12].values
X2 = np.append(arr=np.ones((2400, 1)).astype(int), values = X2, axis = 1)

X_opt2 = X2[:, [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
regressor_ols = sm.OLS(endog= y2, exog= X_opt2).fit()
regressor_ols.summary()


X_ft2 = X_opt2[:, 1:]

from sklearn.model_selection import train_test_split
X2_tr,X2_test,y2_tr,y2_test = train_test_split(X_ft2,y2,test_size=0.2,random_state=0)

from sklearn.ensemble import RandomForestRegressor
reg2 = RandomForestRegressor(n_estimators= 20, criterion="mse", random_state=0)
reg2.fit(X2_tr,y2_tr)

y_pr2 = reg2.predict(X2_test)


testdata = pd.read_csv('test.csv')
TX2 = testdata.iloc[:, [1,2,3,4,5,6,7,8,9,10]].values

ayans2 = reg2.predict(TX2)