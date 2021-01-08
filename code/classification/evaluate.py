import pickle
import os
from pandas import read_csv
from sklearn.metrics import classification_report, confusion_matrix, r2_score
from sklearn.model_selection import train_test_split

TOP_FEATURES = ['valence', 'yearsSinceCreation', 'acousticness', 'danceability',
 'energy', 'explicit', 'instrumentalness', 'liveness', 'loudness', 'tempo', 'popularity_level']

def evaluateModels(x_test, y_test, path):
    for i in os.listdir(path):
        if i.endswith(".sav"):
            print("=================== ", i, " ===================")
            model = pickle.load(open(path+i, 'rb'))
            y_pred = model.predict(x_test)
            print(classification_report(y_test, y_pred))
            print("score : ", model.score(x_test, y_test))
            print("=================== end  ", i," ===================")



def split(songs, testsize):
    X = songs.iloc[:,0:-1]
    Y = songs.iloc[:,-1]
    x_train, x_test, y_train, y_test = train_test_split( X, Y, test_size = testsize,  random_state= 0 )
    return x_train, x_test, y_train, y_test

path = "./models/classificationModels/"

def runEvaluate():
    songs = read_csv("./dataSetCache/classification/songs.csv")
       
    _, x_test, _, y_test = split(songs,0.20)
    
    # Decision Tree
    model = pickle.load(open(path + "dTree.sav", 'rb'))
    name = "Decision Tree"

    print(f"\n\n=================== {name} ===================")
    print("score : ", model.score(x_test, y_test))
    print(f"=================== end {name} ===================\n\n")

    # gaussian Svm
    model = pickle.load(open(path + "gaussianSvm.sav", 'rb'))
    name = "gaussian Svm"

    print(f"\n\n=================== {name} ===================")
    print("score : ", model.score(x_test, y_test))
    print(f"=================== end {name} ===================\n\n")

    # logistic regression
    model = pickle.load(open(path + "logisticReg.sav", 'rb'))
    name = "logistic regression"

    print(f"\n\n=================== {name} ===================")
    print("score : ", model.score(x_test, y_test))
    print(f"=================== end {name} ===================\n\n")
