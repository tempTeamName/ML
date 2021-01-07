import logisticReg
import svm
import pre
import pickle
import time
import matplotlib.pyplot as plt
import dTree
import os
from pandas import DataFrame, read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, r2_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


def split(songs):
    X = songs.iloc[:,0:-1]
    Y = songs["popularity_level"]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20,  random_state= 0 )
    return x_train, x_test, y_train, y_test

def evaluate(model, x_test, y_test):
    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred))
    print("score : ", model.score(x_test, y_test))


def writeTime(strr):
    f = open("./models/time.txt", "a")
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
        model = logisticReg.logisticReg(x_train, y_train,c)
        printStr = f'logisticReg c = {c} time : '+str(time.time() - start_time)
        print(printStr)
        writeTime(printStr)
        pickle.dump(model, open(path+f"logisticReg{c}.sav", 'wb'))


    # SVM linear
    start_time = time.time()
    model = svm.linearSvm(x_train, y_train)
    printStr = 'linear svm time : '+str(time.time() - start_time)
    print(printStr)
    writeTime(printStr)
    pickle.dump(model, open(path+"linearSvm.sav", 'wb'))

    # poly SVM
    degree = [2, 3]
    for deg in degree:
        start_time = time.time()
        model = svm.polySvm(x_train, y_train, deg)
        printStr = f'polySvm degree = {deg} time : '+str(time.time() - start_time)
        print(printStr)
        writeTime(printStr)
        pickle.dump(model, open(path+f"polySvm{deg}.sav", 'wb'))

    # gaussian SVM
    start_time = time.time()
    model = svm.gaussianSvm(x_train, y_train)
    printStr = "gaussianSvm time : "+str(time.time() - start_time)
    print(printStr)
    writeTime(printStr)
    pickle.dump(model, open(path+"gaussianSvm.sav", 'wb'))

    # decision tree
    level = [3, 5, 7]
    for lev in level:
        start_time = time.time()
        model = dTree.dTree(x_train, y_train, lev)
        printStr = f'Decision Tree level = {lev} time : '+str(time.time() - start_time)
        print(printStr)
        writeTime(printStr)
        pickle.dump(model, open(path+f"dTree{lev}.sav", 'wb'))


def evaluateModels(x_test, y_test, path):
    for i in os.listdir(path):
        if i.endswith(".sav"):
            print("=================== ", i)
            model = pickle.load(open(path+i, 'rb'))
            evaluate(model, x_test, y_test)
            print("===end=== ", i)

def evaluateModelsWithTestData(songs: DataFrame):
    _, x_test, _, y_test = split(songs)
    evaluateModels(x_test, y_test, "./models/")
    topFeatures = getTopFeatures(songs, 10)
    topFeatures.append('popularity_level')
    songs = songs[topFeatures]
    _, x_test, _, y_test = split(songs) 
    evaluateModels(x_test, y_test, "./topFeaturesModels/")

def trainWithAllFeatures(songs: DataFrame):
    x_train, x_test, y_train, y_test = split(songs)
    train(x_train, y_train, "./models/")
    return x_test, y_test

def trainWithTopCorrFeatrues(songs: DataFrame):
    topFeatures = getTopFeatures(songs, 10)
    topFeatures.append('popularity_level')
    songs = songs[topFeatures]
    x_train, x_test, y_train, y_test = split(songs)
    train(x_train, y_train, "./topFeaturesModels/")
    return x_test, y_test


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

if __name__ == "__main__":
    songs = read_csv("./dataSetCache/songs.csv")

   
    