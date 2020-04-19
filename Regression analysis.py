# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 12:26:00 2020

@author: skambou
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from scipy.stats.contingency import chi2_contingency
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


#from sklearn.metrics import roc_auc_score, roc_curve

df = pd.read_csv('D:\\Data Science\\ML\\Project\\Regression\\14100092.csv')


df.describe()
df.isnull().sum()
df.info()
df.columns

# Treating NA values

df.drop(['STATUS', 'SYMBOL', 'TERMINATED'], inplace = True, axis = 1)

df.dropna(inplace = True)


corr = df.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(10, 110, n=100),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)


# Extract independent and dependent variables

X = df[['Province_encoding', 'Industry_encoding', 'COORDINATE']]
y = df['VALUE']

from sklearn.utils import shuffle
X, y = shuffle(X, y, random_state = 0)

#splitting data in training and test set
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0)

#Scale the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

'''

REGRESSION ANALYSIS

'''
#create an instance of the model
lin = LinearRegression()

#train the model
lin.fit(X_train, y_train)


#import linear model
import statsmodels.api as sm
X2_train = sm.add_constant(X_train)
X2_test = sm.add_constant(X_test)
ols = sm.OLS(y_train,X2_train)
lr = ols.fit()

while lr.pvalues.max()>0.05:
    X2_train=np.delete(X2_train,lr.pvalues.argmax(),axis=1)
    X2_test=np.delete(X2_test,lr.pvalues.argmax(),axis=1)
    ols = sm.OLS(y_train,X2_train)
    lr = ols.fit()

print(lr.summary())


#coefficient of determination
lin.score(X_train,y_train)
lin.score(X_test,y_test)


#use model to predict
y_predlin = lin.predict(X_test)

test_set_rmse = (np.sqrt(mean_squared_error(y_test, y_predlin)))
test_set_r2 = r2_score(y_test, y_predlin)
adjr2 = 1 - (1-test_set_r2)*((len(X_test) - 1)/(len(X_test) - len(X_test[0])))
print(test_set_rmse, test_set_r2, adjr2)


# Correlation Matrix

features = ['REF_DATE', 'GEO', 'Province_encoding', 'DGUID',
       'North American Industry Classification System (NAICS)',
       'Industry_encoding', 'UOM', 'UOM_ID', 'SCALAR_FACTOR', 'SCALAR_ID',
       'VECTOR', 'COORDINATE', 'VALUE', 'DECIMALS']

mask = np.zeros_like(df[features].corr(), dtype=np.bool) 
mask[np.triu_indices_from(mask)] = True 

f, ax = plt.subplots(figsize=(16, 12))
plt.title('Correlation Matrix',fontsize=25)

sns.heatmap(df[features].corr(),linewidths=0.25,vmax=0.7,square=True,cmap="BuGn", #"BuGn_r" to reverse 
            linecolor='w',annot=True,annot_kws={"size":8},mask=mask,cbar_kws={"shrink": .9})

df.columns

'''
KNN ANALYSIS

'''
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

tuned_parameters = [{'n_neighbors': range(1,10), 'p': range(1,10)}]
knn = GridSearchCV(KNeighborsRegressor(), tuned_parameters, cv=4)
knn.fit(X_train, y_train)
print("Best number of neighbors found {}:".format(knn.best_params_))

#will use K = 4 and p = 1
knn = KNeighborsRegressor(n_neighbors=4, p = 1)
knn.fit(X_train, y_train)
y_predknn = knn.predict(X_test)


test_knn_rmse = (np.sqrt(mean_squared_error(y_test, y_predknn)))
test_knn_r2 = r2_score(y_test, y_predknn)
adjr2_knn = 1 - (1-test_knn_r2)*((len(X_test) - 1)/(len(X_test) - len(X_test[0])))
print(test_knn_rmse, test_knn_r2, adjr2_knn)



"""
ADABOOST ANALYSIS
"""
from sklearn.ensemble import AdaBoostRegressor

tuned_parameters = [{'n_estimators': range(10,110,10)}]
ada = GridSearchCV(AdaBoostRegressor(), tuned_parameters, cv=4)
ada.fit(X_train, y_train)
print("Best number of estimators found {}:".format(ada.best_params_))

#best # estimators is 10

ada = AdaBoostRegressor(n_estimators=10)
ada.fit(X_train, y_train)
ada.score(X_test, y_test)

y_predada = ada.predict(X_test)

test_ada_rmse = (np.sqrt(mean_squared_error(y_test, y_predada)))
test_ada_r2 = r2_score(y_test, y_predada)
adjr2_ada = 1 - (1-test_ada_r2)*((len(X_test) - 1)/(len(X_test) - len(X_test[0])))
print(test_ada_rmse, test_ada_r2, adjr2_ada)




"""
RANDOM FOREST ANALYSIS
"""

from sklearn.ensemble import RandomForestRegressor
tuned_parameters = [{'n_estimators': range(10,110,10)}]
rf = GridSearchCV(RandomForestRegressor(), tuned_parameters, cv=4)
rf.fit(X_train, y_train)
print("Best estimators found {}:".format(rf.best_params_))


tuned_parameters = [{'max_depth': range(1,11)}]
rf = GridSearchCV(RandomForestRegressor(), tuned_parameters, cv=4)
rf.fit(X_train, y_train)
print("Best estimators found {}:".format(rf.best_params_))


tuned_parameters = [{'min_impurity_decrease': range(0,10),
                     'random_state': range(0,60,10)}]
rf = GridSearchCV(RandomForestRegressor(), tuned_parameters, cv=4)
rf.fit(X_train, y_train)
print("Best estimators found {}:".format(rf.best_params_))

rf = RandomForestRegressor(n_estimators=60, max_depth=10, min_impurity_decrease=0, random_state=50)
rf.fit(X_train, y_train)
rf.score(X_test, y_test)

y_predrf = rf.predict(X_test)

test_rf_rmse = (np.sqrt(mean_squared_error(y_test, y_predrf)))
test_rf_r2 = r2_score(y_test, y_predrf)
adjr2_rf = 1 - (1-test_rf_r2)*((len(X_test) - 1)/(len(X_test) - len(X_test[0])))
print(test_rf_rmse, test_rf_r2, adjr2_rf)


"""
SVR ANALYSIS
"""

sc_X = StandardScaler()
X_scaled = sc_X.fit_transform(X)



from sklearn.svm import SVR

#Hyperparameter tuning for gamma

tuned_parameters = [{'kernel': ['linear', 'poly', 'rbf'],
                     'degree' : [2,3],
                     'gamma' : [0.01, 0.1,1,10],
                     'C': [0.01, 0.1,1,10]}]
svm = GridSearchCV(SVR(), tuned_parameters, cv=4)
svm.fit(X_train, y_train)
print("Best estimators found {}:".format(svm.best_params_))


svm = SVR(kernel='rbf', degree=2, gamma=10, C=10)
svm.fit(X_train,y_train)
y_pred_svm=svm.predict(X_test)


test_svm_rmse = (np.sqrt(mean_squared_error(y_test, y_pred_svm)))
test_svm_r2 = r2_score(y_test, y_pred_svm)
adjr2_svm = 1 - (1-test_svm_r2)*((len(X_test) - 1)/(len(X_test) - len(X_test[0])))
print(test_svm_rmse, test_svm_r2, adjr2_svm)





#making comparison for all the models

print('RMSE for Linear Regression :', test_set_rmse)
print('RMSE for KNN :', test_knn_rmse)
print('RMSE for Adaboost :', test_ada_rmse)
print('RMSE for Random Forest :', test_rf_rmse)
print('RMSE for Random Forest :', test_svm_rmse)
print()
print('R2 score for Linear Regression :', test_set_r2*100,'\n'
      'Adj-R2 score for Linear Regression :', adjr2*100)
print('R2 score for KNN :', test_knn_r2*100,'\n'
      'Adj-R2 score for KNN :', adjr2_knn*100)
print('R2 score for Adaboost :', test_ada_r2*100,'\n'
      'Adj-R2 score for Adaboost :', adjr2_ada*100)
print('R2 score for Random Forest :', test_rf_r2*100,'\n'
      'Adj-R2 score for Random Forest :', adjr2_rf*100)
print('R2 score for SVR :', test_svm_r2*100,'\n'
      'Adj-R2 score for SVR :', adjr2_svm*100)

""" Based on the results, we can say that KNN is the current best model for 
this dataset analysis"""


#Kfold cross validation

#preprocessing


sc = StandardScaler()
X_scaled = sc.fit_transform(X)

# Perform 4-fold cross validation
scores = cross_val_score(knn, X_scaled, y, cv=4)
print ('Cross-validated scores:', scores)


