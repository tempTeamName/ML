from pandas import DataFrame
from sklearn.model_selection import train_test_split

def split(songs:DataFrame):
    X = songs.iloc[:,0:-1]
    Y = songs.iloc[:,-1]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20,  random_state= 0 )
    return x_train, x_test, y_train, y_test