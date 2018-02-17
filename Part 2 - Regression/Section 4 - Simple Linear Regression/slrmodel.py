# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 11:11:57 2017

@author: Karthikeyan Sankar
"""
#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing datasets
dataset = pd.read_csv('Salary_Data.csv')

#creating matrix of feautures
x = dataset.iloc[:,:-1].values
#creating dependent vector
y = dataset.iloc[:,1].values

#Splittting the data into test set and train set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

#Fitting the model to train set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#predicting the test set results
y_pred = regressor.predict(x_test)

#Visualising the training set result
plt.scatter(x_train,y_train, color='blue')
plt.plot(x_train,regressor.predict(x_train), color='red')
plt.title('Salary vs Experience')
plt.xlabel('Exp(yrs)')
plt.ylabel('Salary($)')
plt.show()

#visualising the test set results
plt.scatter(x_test,y_test, color='green')
plt.plot(x_train,regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience')
plt.xlabel('Exp(yrs)')
plt.ylabel('Salary($)')
plt.show()




