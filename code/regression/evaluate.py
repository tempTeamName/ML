import os
import pickle
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn import metrics
from pandas import read_csv

path = "./models/regressionModels/"

def getTopCorrFeatures(songs):
    corr = songs.iloc[:,:].corr()
    top_features = corr.index[abs(corr['popularity']) > 0.4]
    return top_features

def split(songs, testsize):
    X = songs.iloc[:,0:-1]
    Y = songs.iloc[:,-1]
    x_train, x_test, y_train, y_test = train_test_split( X, Y, test_size = testsize,  random_state= 0 )
    return x_train, x_test, y_train, y_test

def runEvaluate():

    songs = read_csv("./dataSetCache/regression/songs.csv")
    songsWithArtists = \
        read_csv("./dataSetCache/regression/songsWithAritsts.csv")

    # all featurea but artists     
    _, x_test, _, y_test = split(songs,0.30)
    
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
    _, x_test, _, y_test = split(songsWithArtists,0.30)
    
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

    _, x_test, _, y_test = split(songs[top_features],0.30)

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


def test(songs, songsWithAritsts):

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
    x_test = songsWithAritsts.iloc[:,0:-1]
    y_test = songsWithAritsts.iloc[:,-1]
    
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
