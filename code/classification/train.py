import time
import pickle
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from pandas import DataFrame, read_csv

from classification.dTree import dTree 
from classification.svm import linearSvm, linearSvm, polySvm, gaussianSvm
from classification.logisticReg import logisticReg
from sklearn.model_selection import train_test_split
def split(songs, testsize):
    X = songs.iloc[:,0:-1]
    Y = songs.iloc[:,-1]
    x_train, x_test, y_train, y_test = train_test_split( X, Y, test_size = testsize,  random_state= 0 )
    return x_train, x_test, y_train, y_test

def writeTime(strr, path):
    f = open(path+"time.txt", "a")
    f.write(strr+'\n')
    f.close()

def getTopFeatures(songs: DataFrame, K: int):
    X = songs.iloc[:,0:-1]
    y = songs["popularity_level"]
    fs = SelectKBest(score_func=f_classif, k=K)
    fs.fit_transform(X, y)
    mask = fs.get_support()
    features = []
    for i in range(len(songs.columns)-1):
        if(mask[i] == True):
            features.append(songs.columns[i])
    return features

path = "./models/classificationModels/"

def runTrain():
    songs = read_csv("./dataSetCache/classification/songs.csv")

    # clear the time file
    f = open(path+"time.txt", "w")
    f.close()

    x_train, _, y_train, _ = split(songs,0.20)

    # Decision Tree
    lev = 5

    start_time = time.time()
    model = dTree(x_train, y_train, lev)
    printStr = 'Decision Tree time : '+str(time.time() - start_time)
    print(printStr)
    writeTime(printStr, path)
    pickle.dump(model, open(path+"dTree.sav", 'wb'))
    
    # gaussian Svm
    start_time = time.time()
    model = gaussianSvm(x_train, y_train)
    printStr = "gaussian Svm time : "+str(time.time() - start_time)
    print(printStr)
    writeTime(printStr, path)
    pickle.dump(model, open(path+"gaussianSvm.sav", 'wb'))

    
    # logistic regression
    
    c = 10

    start_time = time.time()
    model = logisticReg(x_train, y_train, c)
    printStr = 'logisticReg time : '+str(time.time() - start_time)
    print(printStr)
    writeTime(printStr, path)
    pickle.dump(model, open(path+"logisticReg.sav", 'wb'))
