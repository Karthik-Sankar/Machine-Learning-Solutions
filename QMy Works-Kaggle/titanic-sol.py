# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 12:11:02 2018

@author: Karthikeyan Sankar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('train.csv')

X = dataset.iloc[:, [2,4,5,6,7,9,11]].values
y = dataset.iloc[:, 1].values

#Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
X[:,[2]] = imputer.fit_transform(X[:,[2]])

#string data missing
m_c = pd.get_dummies(X[:,-1]).sum().sort_values(ascending=False).index[0]
dv = pd.Series(X[:,-1])                   
def ret_freq(x):
    if((pd.isnull(x))):
        return m_c
    else:
        return x
X[:,-1] = dv.map(ret_freq)

#Dealing with the categorical data
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
X[:, 1] = enc.fit_transform(X[:,1])

enc2 = LabelEncoder()
X[:, -1] = enc.fit_transform(X[:,-1])

#taking care of dummy variable trap
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[6])
X = ohe.fit_transform(X).toarray()
X = X[:, 1:]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                      random_state=0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)
X_train = lda.fit_transform(X_train,y_train)
X_test = lda.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
classifier = SVC(C=100, kernel='rbf', random_state=0)
#classifier = LogisticRegression()
#classifier = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

cm_score = (cm[0][0]+cm[1][1])/len(X_test)


#using k fold 
from sklearn.model_selection import cross_val_score
cvs = cross_val_score(classifier,X_train,y_train, cv=10)
cvs.mean()
cvs.std()

#using grid search
from sklearn.model_selection import GridSearchCV
parameters = [{'C':[1,10,100,1000], 'kernel':['linear']},
               {'C':[1,10,100,1000], 'kernel':['rbf'], 
               'gamma':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}]

gscv = GridSearchCV(estimator=classifier, param_grid=parameters,
                    scoring='accuracy', cv=10, n_jobs=1)
gscv = gscv.fit(X_train,y_train)
gscv.best_score_
gscv.best_params_

from sklearn.ensemble import RandomForestClassifier
clas= RandomForestClassifier()