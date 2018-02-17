import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, [0,2,4,5,6,7,9,11]].values
y = dataset.iloc[:,1].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:,3:4])
X[:,3:4]  = imputer.transform(X[:,3:4])

from sklearn.preprocessing import LabelEncoder
labelenc = LabelEncoder()
X[:, 2] = labelenc.fit_transform(X[:,2])

m_c = pd.get_dummies(X[:,-1]).sum().sort_values(ascending=False).index[0]
dv = pd.Series(X[:,-1])

def ret_mfreq(x):
    if(pd.isnull(x)):
        return m_c
    else:
        return x

X[:,-1] = dv.map(ret_mfreq)

from sklearn.preprocessing import LabelEncoder
labelenc = LabelEncoder()
X[:, -1] = labelenc.fit_transform(X[:,-1])


from sklearn.preprocessing import OneHotEncoder
oneHE = OneHotEncoder(categorical_features=[7])
X = oneHE.fit_transform(X).toarray()

X = X[:,1:]

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(X_train)
x_test = sc_x.fit_transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)



import statsmodels.formula.api as sm
X = np.append(arr=np.ones((891,1)).astype(int), values=X, axis=1)
X_opt = X[:,[0,1,2,3,4,5,6,7,8,9]]
regressor_sm = sm.OLS(endog=y, exog=X_opt).fit()
regressor_sm.summary()

X_opt = X[:,[0,1,3,4,5,6,7,8,9]]
regressor_sm = sm.OLS(endog=y, exog=X_opt).fit()
regressor_sm.summary()

X_opt = X[:,[0,1,3,5,6,7,8,9]]
regressor_sm = sm.OLS(endog=y, exog=X_opt).fit()
regressor_sm.summary()

X_opt = X[:,[0,1,3,5,6,7,8]]
regressor_sm = sm.OLS(endog=y, exog=X_opt).fit()
regressor_sm.summary()

Xo_train,Xo_test,yo_train,yo_test = train_test_split(X_opt,y,test_size=0.2,random_state=0)


from sklearn.linear_model import LogisticRegression
newclass = LogisticRegression()
newclass.fit(Xo_train,yo_train)


yo_pred = newclass.predict(Xo_test)
cm2 = confusion_matrix(yo_test,yo_pred)



dataset2 = pd.read_csv('train.csv')
Xt = dataset2.iloc[:, [0,1,3,4,5,6,8,10]].values


#imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
#imputer = imputer.fit(Xt[:,3:4])
#Xt[:,3:4]  = imputer.transform(Xt[:,3:4])
