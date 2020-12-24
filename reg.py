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

def reg(**parms):
    '''
    Parameters
    ----------
    songs: dataFrame
        input dataFrame.
    isTopFeatures: bool
        select top features if True.
    degree: num
        Polynomial Features degree
    alpha: num
        alpha for the liner nodel
    normalize: bool 
        
    '''
    songs = parms["songs"]
    # get top features
    if parms["isTopFeatures"]:
        corr = songs.iloc[:,:].corr()
        top_features = corr.index[abs(corr['popularity']) > 0.4]
        songs = songs[top_features]

    # extract X and Y
    X = songs.iloc[:,0:-1]
    Y = songs["popularity"]

    # data scaling
    # X[:] = pd.DataFrame(preprocessing.scale(X[:]))

    # split data 
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30, random_state = 0 )
    
    # PolynomialFeatures
    poly_features = PolynomialFeatures(degree=parms["degree"])
    X_train_poly = poly_features.fit_transform(X_train)

    # linear regression 
    poly_model = linear_model.Ridge(alpha=parms["alpha"], normalize=parms["normalize"])
    poly_model.fit(X_train_poly, y_train)

    # testing 
    y_train_predicted = poly_model.predict(X_train_poly)
    prediction = poly_model.predict(poly_features.fit_transform(X_test))

    true_value=np.asarray(y_test)[0]
    predicted_value=prediction[0]

    print('Co-efficients len : ',len(poly_model.coef_))
    print("Co-efficients max :", max(poly_model.coef_))
    print("Co-efficients min :", min(poly_model.coef_))
    print('Intercept :%.3f'%poly_model.intercept_)
    print('MSE :%.3f'%metrics.mean_squared_error(y_test, prediction))
    print('r2 :%.3f'%r2_score(y_test, prediction))