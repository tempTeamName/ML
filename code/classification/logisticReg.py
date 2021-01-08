from sklearn.linear_model import LogisticRegression

def logisticReg(x_train, y_train, c):
    logisticRegression = LogisticRegression(multi_class='ovr', solver='liblinear', C=c)
    model = logisticRegression.fit(x_train, y_train)
    return model