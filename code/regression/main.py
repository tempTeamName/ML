import reg
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import time
from pandas import read_csv
from pre import pre
import time


def test(songs):
    degree = [1,2,3,4]
    alpha = [0.01,0.1,0.5,1,10]
    for degreeI in degree:
        for alphaI in alpha:
            print("degree :", degreeI, "\nalpha :",alphaI,"\n\n")
            reg.reg(songs = songs,isTopFeatures= False, degree = degreeI , alpha = alphaI, normalize = True)
            print("\n\n==================================") 

def getTopCorrFeatures(songs):
    corr = songs.iloc[:,:].corr()
    top_features = corr.index[abs(corr['popularity']) > 0.4]
    return top_features


def train(x_train, y_train, path: str, top: bool):

    # clear the time file
    f = open(path+"time.txt", "w")
    f.close()

    if top:
        # with top correlated features 

        deg = 4
        alpha = 0.01

        start_time = time.time()
        model = reg(x_train, y_train, deg, alpha)
        printStr = f'regression degree = {deg} time : '+str(time.time() - start_time)
        print(printStr)
        writeTime(printStr, path)
        pickle.dump(model, open(path+f"{deg}_{alpha}_regressionTOP.sav", 'wb'))


    deg = 3
    alpha = 0.01

    start_time = time.time()
    model = reg(x_train, y_train, deg, alpha)
    printStr = f'regression degree = {deg} time : '+str(time.time() - start_time)
    print(printStr)
    writeTime(printStr, path)
    pickle.dump(model, open(path+f"{deg}_{alpha}_regressionWithAll.sav", 'wb'))


    deg = 3
    alpha = 0.5

    start_time = time.time()
    model = reg(x_train, y_train, deg, alpha)
    printStr = f'regression degree = {deg} time : '+str(time.time() - start_time)
    print(printStr)
    writeTime(printStr, path)
    pickle.dump(model, open(path+f"{deg}_{alpha}_regressionWithAll.sav", 'wb'))


    # reg with cv
    cv = 10

    start_time = time.time()
    model = regWithCV(x_train, y_train, cv)
    printStr = f'regression with cv = 10 time : '+str(time.time() - start_time)
    print(printStr)
    writeTime(printStr, path)
    pickle.dump(model, open(path+f"{deg}_{alpha}_regressionCV.sav", 'wb'))

if __name__ == "__main__":
    songs = read_csv("spotify_training.csv")
    print(songs.decribe())
    songs1, songs2 = pre(songs)
       
    start_time = time.time()
    print("======== all selected features with degree 3 and alpha 0.01 ========")
    reg.regression_with_Ridge(songs = songs2, degree = 3 , alpha = 0.01)
    print('execution time ',(time.time() - start_time))
    start_time = time.time()
    print("\n\n======== all selected features except artists with degree 4 and anlpha 0.01 (current best fit) ========")
    reg.regression_with_Ridge(songs = songs1, degree = 4 , alpha = 0.01)
    print('execution time ',(time.time() - start_time))
    start_time = time.time()
    print("\n\n======== top corr features with degree 3 and anlpha 0.01 ========")
    reg.regression_with_TopFeatures(songs = songs1, degree = 3 , alpha = 0.01)
    print('execution time ',(time.time() - start_time))
    start_time = time.time()
    print("\n\n======== top corr features with degree 4 and anlpha 0.01 ========")
    reg.regression_with_TopFeatures(songs = songs1, degree = 4 , alpha = 0.01)
    print('execution time ',(time.time() - start_time))
    start_time = time.time()
    print("\n\n======== regression with cv k=10 ======== ")
    reg.regression_with_cv(songs= songs1, degree = 4 , alpha = 0.01 )
    print('execution time ',(time.time() - start_time))

# best res with  dg = 4 , alpha : 0.01, all selected features