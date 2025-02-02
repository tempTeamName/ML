import os
import time
import pickle
from pandas import DataFrame, read_csv
from regression.reg import reg
from sklearn.model_selection import train_test_split

def split(songs, testsize):
    X = songs.iloc[:,0:-1]
    Y = songs.iloc[:,-1]
    x_train, x_test, y_train, y_test = train_test_split( X, Y, test_size = testsize,  random_state= 0 )
    return x_train, x_test, y_train, y_test

path = "./models/regressionModels/"

def writeTime(strr, path):
    f = open(path+"time.txt", "a")
    f.write(strr+'\n')
    f.close()


def getTopCorrFeatures(songs):
    corr = songs.iloc[:,:].corr()
    top_features = corr.index[abs(corr['popularity']) > 0.4]
    return top_features


def runTrain():
    songs = read_csv("./dataSetCache/regression/songs.csv")
    songsWithArtists = \
        read_csv("./dataSetCache/regression/songsWithAritsts.csv")

    f = open(path+"time.txt", "w")
    f.close()
    
    # train with all featurea but artists
    x_train, _, y_train, _ = split(songs,0.30)

    deg = 4
    alpha = 0.01

    start_time = time.time()
    model = reg(x_train, y_train, deg, alpha)
    printStr = 'regression time : '+str(time.time() - start_time)
    print(printStr)
    writeTime(printStr, path)
    pickle.dump(model, open(path + "regression.sav", 'wb'))


    # train with all featurea
    x_train, _, y_train, _ = split(songsWithArtists,0.30)
    
    deg = 3
    alpha = 0.01

    start_time = time.time()
    model = reg(x_train, y_train, deg, alpha)
    printStr = 'regressionWithArtists time : '\
        +str(time.time() - start_time)
    print(printStr)
    writeTime(printStr, path)
    pickle.dump(model, open(path + "regressionWithArtists.sav", 'wb'))


    # train with the top correlated features
    songs = songs[getTopCorrFeatures(songs)]
    x_train, _, y_train, _ = split(songs,0.30)
   
    deg = 4
    alpha = 0.01

    start_time = time.time()
    model = reg(x_train, y_train, deg, alpha)
    printStr = 'regressionTOP time : '+str(time.time() - start_time)
    print(printStr)
    writeTime(printStr, path)
    pickle.dump(model, open(path + "regressionTOP.sav", 'wb'))
