import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def dTree(x_train, y_train, maxd):
    dTree = DecisionTreeClassifier(max_depth=maxd)
    model = dTree.fit(x_train, y_train)
    return model