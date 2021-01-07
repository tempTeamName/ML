import pickle
import os
from pandas import read_csv
from sklearn.metrics import classification_report, confusion_matrix, r2_score

TOP_FEATURES = ['valence', 'yearsSinceCreation', 'acousticness', 'danceability',
 'energy', 'explicit', 'instrumentalness', 'liveness', 'loudness', 'tempo', 'popularity_level']

def evaluateModels(x_test, y_test, path):
    for i in os.listdir(path):
        if i.endswith(".sav"):
            print("=================== ", i)
            model = pickle.load(open(path+i, 'rb'))
            y_pred = model.predict(x_test)
            print(classification_report(y_test, y_pred))
            print("score : ", model.score(x_test, y_test))
            print("===end=== ", i)


def runEvaluate(songs):

    X = songs.iloc[:,0:-1]
    Y = songs.iloc[:,-1]
    evaluateModels(X, Y, "./models/classificationModels/models/")
    
    X = songs[TOP_FEATURES].iloc[:,0:-1]
    Y = songs[TOP_FEATURES].iloc[:,-1]
    evaluateModels(X, Y, "./models/classificationModels/topFeaturesModels/")
