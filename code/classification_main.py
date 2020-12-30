import logisticReg
import svm
import pre
import pickle
import time
import matplotlib.pyplot as plt
import dTree
from pandas import DataFrame, read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, r2_score


def split(songs):
    X = songs.iloc[:,0:-1]
    Y = songs["popularity_level"]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20,  random_state= 0 )
    return x_train, x_test, y_train, y_test

def evaluate(model, x_test, y_test):
    y_pred = model.predict(x_test)
    print(model.classes_)
    print(classification_report(y_test, y_pred))
    print("score :", model.score(x_test, y_test))
    print('Co-efficients : ',model.coef_)
    print('Intercept :',model.intercept_)

    cm = confusion_matrix(y_test, y_pred)
    _, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cm)
    ax.grid(False)
    ax.xaxis.set(ticks=(0, 1, 2), ticklabels=('Predicted High', 'Predicted Intermediate', 'Predicted Low'))
    ax.yaxis.set(ticks=(0, 1, 2), ticklabels=('Actual High', 'Actual Intermediate', "Actual LOW"))
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
    plt.show()
    # print('MSE :%.3f'%metrics.mean_squared_error(y_test, prediction))
    # print('MAE :%.3f'%metrics.mean_absolute_error(y_test, prediction))


def writeTime(strr):
    f = open("time.txt", "a")
    f.write(strr+'\n')
    f.close()


def train(x_train, y_train):
    path = "./models/"
    start_time = time.time()
    model = logisticReg.logisticReg(x_train, y_train,1)
    print('logisticReg c = 1 time : ',(time.time() - start_time))
    writeTime('logisticReg c = 1 time : '+str(time.time() - start_time))
    pickle.dump(model, open(path+"logisticReg.sav", 'wb'))

    start_time = time.time()
    model = logisticReg.logisticReg(x_train, y_train,10)
    print('logisticReg c = 10 time : ',(time.time() - start_time))
    writeTime('logisticReg c = 10 time : '+str(time.time() - start_time))
    pickle.dump(model, open(path+"logisticReg.sav", 'wb'))

    start_time = time.time()
    model = logisticReg.logisticReg(x_train, y_train,100)
    print('logisticReg c = 100 time : ',(time.time() - start_time))
    writeTime('logisticReg c = 100 time : '+str(time.time() - start_time))
    pickle.dump(model, open(path+"logisticReg.sav", 'wb'))
    
    start_time = time.time()
    model = svm.linearSvm(x_train, y_train)
    print("linear svm time : ",(time.time() - start_time))
    writeTime('linear svm time : '+str(time.time() - start_time))
    pickle.dump(model, open(path+"linearSvm.sav", 'wb'))
    
    start_time = time.time()
    model = svm.polySvm(x_train, y_train, 2)
    print("polySvm2 time : ",(time.time() - start_time))
    writeTime('polySvm2 time : '+str(time.time() - start_time))
    pickle.dump(model, open(path+"polySvm2.sav", 'wb'))

    start_time = time.time()
    model = svm.polySvm(x_train, y_train, 3)
    print("polySvm3 time : ",(time.time() - start_time))
    writeTime('polySvm3 time : '+str(time.time() - start_time))
    pickle.dump(model, open(path+"polySvm3.sav", 'wb'))
    
    start_time = time.time()
    model = svm.polySvm(x_train, y_train, 4)
    print("polySvm4 time : ",(time.time() - start_time))
    writeTime('polySvm4 time : '+str(time.time() - start_time))
    pickle.dump(model, open(path+"polySvm4.sav", 'wb'))

    start_time = time.time()
    model = svm.gaussianSvm(x_train, y_train)
    print("gaussianSvm time : ",(time.time() - start_time))
    writeTime('polySvm4 time : '+str(time.time() - start_time))
    pickle.dump(model, open(path+"gaussianSvm.sav", 'wb'))

    start_time = time.time()
    model = dTree.dTree(x_train, y_train, 3)
    print("Decision Tree time : ",(time.time() - start_time))
    writeTime('Decision Tree time : '+str(time.time() - start_time))
    pickle.dump(model, open(path+"dTree.sav", 'wb'))


def getModels():
    logRegModel = pickle.load(open("logisticReg.sav", 'rb'))
    linearSvmModel = pickle.load(open("linearSvm.sav", 'rb'))
    polySvm2Model = pickle.load(open("polySvm2.sav", 'rb'))
    polySvm3Model = pickle.load(open("polySvm3.sav", 'rb'))
    polySvm4Model = pickle.load(open("polySvm4.sav", 'rb'))
    gaussianSvmModel = pickle.load(open("gaussianSvm.sav", 'rb'))
    
    return {
        "logisticReg": logRegModel,
        "linearSvm": linearSvmModel,
        "polySvm2": polySvm2Model,
        "polySvm3":polySvm3Model,
        "polySvm4":polySvm4Model,
        "gaussianSvm": gaussianSvmModel
    }
    


if __name__ == "__main__":
    # songs = read_csv("../datasets/spotify_training_classification.csv")
    # songs,songsWithArtists = pre.pre(songs)
    # songs = read_csv("songs.csv")
    songsWithArtists = read_csv("songsWithArtists.csv")
    x_train, x_test, y_train, y_test = split(songsWithArtists)
    train(x_train, y_train)