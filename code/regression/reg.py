import pandas as pd 
import numpy as np 
import math
import seaborn as sns
import matplotlib.pyplot as plt
from random import randrange
import random
import string
import category_encoders as ce 
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import shuffle
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

def reg(x_train, y_train, degree, alpha):
    # PolynomialFeatures
    poly_features = PolynomialFeatures(degree=degree)
    X_train_poly = poly_features.fit_transform(x_train)

    # regression 
    poly_model = linear_model.Ridge(alpha=alpha, normalize = True)
    return poly_model.fit(X_train_poly, y_train)


def regression_with_Ridge(**parms):
    '''
    Parameters
    ---------- 
    songs: dataFrame
        input dataFrame.
    degree: num
        Polynomial Features degree
    alpha: num
        alpha for the liner nodel
    '''
    songs = parms['songs'];
    # extract X and Y
    X = songs.iloc[:,0:-1]
    Y = songs["popularity"]

    # split data 
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30, random_state = 0 )
    
    # PolynomialFeatures
    poly_features = PolynomialFeatures(degree=parms["degree"])
    X_train_poly = poly_features.fit_transform(X_train)

    # linear regression 
    poly_model = linear_model.Ridge(alpha=parms["alpha"], normalize=True)
    poly_model.fit(X_train_poly, y_train)

    prediction = poly_model.predict(poly_features.fit_transform(X_test))

    print('Co-efficients len : ',len(poly_model.coef_))
    print("Co-efficients max :", max(poly_model.coef_))
    print("Co-efficients min :", min(poly_model.coef_))
    print('Intercept :%.3f'%poly_model.intercept_)
    print('MSE :%.3f'%metrics.mean_squared_error(y_test, prediction))
    print('MAE :%.3f'%metrics.mean_absolute_error(y_test, prediction))
    print('r2 :%.3f'%r2_score(y_test, prediction))


def regression_with_cv(**parms):
    '''
    Parameters
    ----------
    songs: dataFrame
        input dataFrame.
    degree: num
        Polynomial Features degree
    alpha: num
        alpha for the liner nodel
    '''
    songs = parms["songs"]
     # extract X and Y
    X = songs.iloc[:,0:-1]
    Y = songs["popularity"]

    # split data 
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30, random_state = 0 )
    
    # PolynomialFeatures
    poly_features = PolynomialFeatures(degree=parms["degree"])
    X_train_poly = poly_features.fit_transform(X_train)

    # linear regression 
    poly_model = linear_model.Ridge(alpha=parms["alpha"], normalize=True)
    poly_model.fit(X_train_poly, y_train)

    cvModel = linear_model.LinearRegression()
    cv_neg_mse = cross_val_score(cvModel, X_train, y_train, scoring='neg_mean_squared_error', cv=10)
    cv_r2 = cross_val_score(cvModel, X_train, y_train, scoring='r2', cv=10)
    print("cv neg MSE :",cv_neg_mse.mean())
    print("cv r2 :",cv_r2.mean())
    
    # testing 
    prediction = poly_model.predict(poly_features.fit_transform(X_test))

    print('Co-efficients len : ',len(poly_model.coef_))
    print("Co-efficients max :", max(poly_model.coef_))
    print("Co-efficients min :", min(poly_model.coef_))
    print('Intercept :%.3f'%poly_model.intercept_)
    print('MSE :%.3f'%metrics.mean_squared_error(y_test, prediction))
    print('MAE :%.3f'%metrics.mean_absolute_error(y_test, prediction))
    print('r2 :%.3f'%r2_score(y_test, prediction))


def regression_with_TopFeatures(**parms):
    '''
    Parameters
    ----------
    songs: dataFrame
        input dataFrame.
    degree: num
        Polynomial Features degree
    alpha: num
        alpha for the liner nodel
    '''
    songs = parms["songs"]
    songs = songs[getTopCorrFeatures(songs)]
    # extract X and Y
    X = songs.iloc[:,0:-1]
    Y = songs["popularity"]

    # split data 
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30, random_state = 0 )
    
    # PolynomialFeatures
    poly_features = PolynomialFeatures(degree=parms["degree"])
    X_train_poly = poly_features.fit_transform(X_train)

    # linear regression 
    poly_model = linear_model.Ridge(alpha=parms["alpha"], normalize=True)
    poly_model.fit(X_train_poly, y_train)

     # testing 
    prediction = poly_model.predict(poly_features.fit_transform(X_test))

    print('Co-efficients len : ',len(poly_model.coef_))
    print("Co-efficients max :", max(poly_model.coef_))
    print("Co-efficients min :", min(poly_model.coef_))
    print('Intercept :%.3f'%poly_model.intercept_)
    print('MSE :%.3f'%metrics.mean_squared_error(y_test, prediction))
    print('MAE :%.3f'%metrics.mean_absolute_error(y_test, prediction))
    print('r2 :%.3f'%r2_score(y_test, prediction))
