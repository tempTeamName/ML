import time
import pickle
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from pandas import DataFrame, read_csv

from dTree import dTree 
from svm import linearSvm, linearSvm, polySvm, gaussianSvm
from logisticReg import logisticReg
from sklearn.model_selection import train_test_split


def split(songs, testsize):
    X = songs.iloc[:,0:-1]
    Y = songs.iloc[:,-1]
    x_train, x_test, y_train, y_test = train_test_split( X, Y, test_size = testsize,  random_state= 0 )
    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = split(read_csv("./code/classification/songs.csv"), 0.2)

    # logisticReg
    C = [1, 10, 100]
    for c in C:   
        start_time = time.time()
        model = logisticReg(x_train, y_train,c)
        printStr = f'logisticReg c = {c} time : '+str(time.time() - start_time)
        print(printStr)
        print("score : ", model.score(x_test, y_test))

    # decision tree
    level = [3, 5, 7]
    for lev in level:
        start_time = time.time()
        model = dTree(x_train, y_train, lev)
        printStr = f'Decision Tree level = {lev} time : '+str(time.time() - start_time)
        print(printStr)
        print("score : ", model.score(x_test, y_test))

    # poly SVM
    degree = [2, 3]
    for deg in degree:
        start_time = time.time()
        model = polySvm(x_train, y_train, deg)
        printStr = f'polySvm degree = {deg} time : '+str(time.time() - start_time)
        print(printStr)
        print("score : ", model.score(x_test, y_test))


    

