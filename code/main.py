import pandas
import os, shutil
from pandas import read_csv
from classification.train import runTrain as classificationTrain
from classification.evaluate import runEvaluate as classificationEvaluate, test as classificationTest
from regression.train import runTrain as regressionTrain 
from regression.evaluate import runEvaluate as regressionEvaluate, test as regressionTest
from preprocessing.pre import runPre, preForNew

def clearFolder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def createFolder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def clearTmpData():
    clearFolder("./dataSetCache/")
    clearFolder("./models/")
    init()

def testNew():
    songsRegression = read_csv("./testing/spotify_testing.csv")[:49656]
    songsClassification = read_csv("./testing/spotify_testing_classification.csv")

    # print("songs regression")
    # print(songsRegression.describe())
    # print(songsRegression)

    # print("songs classification")
    # print(songsClassification.describe())

     # test the new dataset in regression
    # print("songs regression")
    # print(songs.describe())
    # print(songs)

    songs, songsWithAritsts = preForNew(songsRegression)
    songs.to_csv("./testing/dataSetCache/regression/songs.csv", index=False)
    songsWithAritsts.to_csv("./testing/dataSetCache/regression/songsWithAritsts.csv", index=False)
    regressionTest(songs, songsWithAritsts)

    # test the new dataset in classification
    songs, songsWithAritsts = preForNew(songsClassification)
    songs.to_csv("./testing/dataSetCache/classification/songs.csv", index=False)
    songsWithAritsts.to_csv("./testing/dataSetCache/classification/songsWithAritsts.csv", index=False)
    classificationTest(songs)

def init():
    createFolder("./dataSetCache/")
    createFolder("./models/")

    createFolder("./dataSetCache/classification")
    createFolder("./dataSetCache/regression")
    
    createFolder("./models/classificationModels")
    createFolder("./models/regressionModels")

    createFolder("./testing/dataSetCache/classification")
    createFolder("./testing/dataSetCache/regression")

def main():
    init()
    print("1. for preprocessing")
    print("2. to train the models")
    print("3. to evaluate the models with the split test")
    print("4. to predict (put the testing files (.csv) in testing floder ) ")
    print("5. for clear all tmp data and models")
    print("6. for full run")
    ch = input()
    if(ch == '1'):
        runPre()
    if(ch == '2'):
        regressionTrain()
        classificationTrain()
    if(ch == '3'):
        regressionEvaluate()
        classificationEvaluate()
    if(ch == '4'):
        testNew()
    if(ch == '5'):
        yes = input("are you sure (yes/no)?")
        if yes == 'yes':
            clearTmpData()
    if(ch == '6'):
        runPre()
        regressionTrain()
        classificationTrain()
        regressionEvaluate()
        classificationEvaluate()
        testNew()

if __name__ == "__main__":
    while True:
        main()