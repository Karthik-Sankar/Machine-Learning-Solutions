# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 22:46:47 2017

@author: Karthikeyan Sankar
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 19:16:20 2017

@author: Karthikeyan Sankar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Social_Network_Ads.csv')

X = dataset.iloc[:, 2:-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_trian, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)
X_train = lda.fit_transform(X_train,y_trian)
X_test = lda.transform(X_test)
#explained_variance = pca.explained_variance_ratio_

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_trian)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm_per = str(((cm[0][0]+cm[1][1])/len(X_test))*100)+'% Accurate'

               
from sklearn.model_selection import cross_val_score
cvs = cross_val_score(estimator=classifier, X=X_train, y=y_trian, cv=10)
cvs.mean()
cvs.std()
