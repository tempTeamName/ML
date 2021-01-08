import os
import pickle
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from pandas import read_csv

path = "./models/regressionModels/"

def getTopCorrFeatures(songs):
    corr = songs.iloc[:,:].corr()
    top_features = corr.index[abs(corr['popularity']) > 0.4]
    return top_features

def runEvaluate():

    songs = read_csv("./dataSetCache/regression/songs.csv")
    songsWithArtists = \
        read_csv("./dataSetCache/regression/songsWithAritsts.csv")

    # all featurea but artists     
    x_test = songs.iloc[:,0:-1]
    y_test = songs.iloc[:,-1]
    
    model = pickle.load(open(path + "regression.sav", 'rb'))
    degree = 4
    name = "regression without artists"

    print(f"\n\n=================== {name} ===================")
    poly_features = PolynomialFeatures(degree=degree)
    y_pred = model.predict(poly_features.fit_transform(x_test))
    print('MSE :%.3f'%metrics.mean_squared_error(y_test, y_pred))
    print('MAE :%.3f'%metrics.mean_absolute_error(y_test, y_pred))
    print('r2 :%.3f'%metrics.r2_score(y_test, y_pred))
    print(f"=================== end {name} ===================\n\n")

    # all featurea 
    x_test = songsWithArtists.iloc[:,0:-1]
    y_test = songsWithArtists.iloc[:,-1]
    
    model = pickle.load(open(path + "regressionWithArtists.sav", 'rb'))
    degree = 3
    name = "regression with artists"

    print(f"\n\n=================== {name} ===================")
    poly_features = PolynomialFeatures(degree=degree)
    y_pred = model.predict(poly_features.fit_transform(x_test))
    print('MSE :%.3f'%metrics.mean_squared_error(y_test, y_pred))
    print('MAE :%.3f'%metrics.mean_absolute_error(y_test, y_pred))
    print('r2 :%.3f'%metrics.r2_score(y_test, y_pred))
    print(f"=================== end {name} ===================\n\n")
    

    # with top correlated features
    top_features = getTopCorrFeatures(songs)

    x_test = songs[top_features].iloc[:,0:-1]
    y_test = songs[top_features].iloc[:,-1]

    model = pickle.load(open(path + "regressionTOP.sav", 'rb'))
    degree = 4
    name = "regression with top correlated features"

    print(f"\n\n=================== {name} ===================")
    poly_features = PolynomialFeatures(degree=degree)
    y_pred = model.predict(poly_features.fit_transform(x_test))
    print('MSE :%.3f'%metrics.mean_squared_error(y_test, y_pred))
    print('MAE :%.3f'%metrics.mean_absolute_error(y_test, y_pred))
    print('r2 :%.3f'%metrics.r2_score(y_test, y_pred))
    print(f"=================== end {name} ===================\n\n")
