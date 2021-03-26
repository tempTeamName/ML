from sklearn.svm import SVC
from sklearn import preprocessing

def polySvm(x_train, y_train, degree):
    
    svclassifier = SVC(kernel='poly', degree=degree)
    model = svclassifier.fit(x_train, y_train)
    return model

def linearSvm(x_train, y_train):

    svclassifier = SVC(kernel='linear')
    model = svclassifier.fit(x_train, y_train)
    return model

def gaussianSvm(x_train, y_train):
    svclassifier = SVC(kernel='rbf')
    model = svclassifier.fit(x_train, y_train)
    return model