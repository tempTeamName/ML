import time
import pickle
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from pandas import DataFrame, read_csv

from classification.dTree import dTree 
from classification.svm import linearSvm, linearSvm, polySvm, gaussianSvm
from classification.logisticReg import logisticReg
from preprocessing.utilities import split


def writeTime(strr, path):
    f = open(path+"time.txt", "a")
    f.write(strr+'\n')
    f.close()

def train(x_train, y_train, path):

    # clear the time file
    f = open(path+"time.txt", "w")
    f.close()

    # logisticReg
    C = [1, 10, 100]
    for c in C:   
        start_time = time.time()
        model = logisticReg(x_train, y_train,c)
        printStr = f'logisticReg c = {c} time : '+str(time.time() - start_time)
        print(printStr)
        writeTime(printStr, path)
        pickle.dump(model, open(path+f"logisticReg{c}.sav", 'wb'))


    # SVM linear
    start_time = time.time()
    model = linearSvm(x_train, y_train)
    printStr = 'linear svm time : '+str(time.time() - start_time)
    print(printStr)
    writeTime(printStr, path)
    pickle.dump(model, open(path+"linearSvm.sav", 'wb'))

    # poly SVM
    degree = [2, 3]
    for deg in degree:
        start_time = time.time()
        model = polySvm(x_train, y_train, deg)
        printStr = f'polySvm degree = {deg} time : '+str(time.time() - start_time)
        print(printStr)
        writeTime(printStr, path)
        pickle.dump(model, open(path+f"polySvm{deg}.sav", 'wb'))

    # gaussian SVM
    start_time = time.time()
    model = gaussianSvm(x_train, y_train)
    printStr = "gaussianSvm time : "+str(time.time() - start_time)
    print(printStr)
    writeTime(printStr, path)
    pickle.dump(model, open(path+"gaussianSvm.sav", 'wb'))

    # decision tree
    level = [3, 5, 7]
    for lev in level:
        start_time = time.time()
        model = dTree(x_train, y_train, lev)
        printStr = f'Decision Tree level = {lev} time : '+str(time.time() - start_time)
        print(printStr)
        writeTime(printStr, path)
        pickle.dump(model, open(path+f"dTree{lev}.sav", 'wb'))


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

def runTrain(songs:DataFrame):
    if songs is None:
        songs = read_csv("dataSetCache/songs.csv")

    # train with all featurea but artists
    x_train, _, y_train, _ = split(songs)
    train(x_train, y_train, "./models/classificationModels/models")

    # train with the top 10 features
    topFeatures = getTopFeatures(songs, 10)
    topFeatures.append('popularity_level')
    songs = songs[topFeatures]
    x_train, _, y_train, _ = split(songs)
    train(x_train, y_train, "./models/classificationModels/topFeaturesModels/")
