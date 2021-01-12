import pandas as pd 
import numpy as np 
import math
from random import randrange
import random
import string
import category_encoders as ce 
import pickle
from pandas import read_csv, DataFrame
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

def cleanArtists(songs, y_name, path):
    '''drop rows with invalid values and destringify the list of artists'''
    songs['artists'] = songs['artists'].apply(lambda x: x[1:-1].split(', ') if(type(x) == str and len(x)) else [])
    songs['artists'] = songs['artists'].apply(lambda x: list(map(lambda y: y[1:-1], x)) )
    encoder = ce.HashingEncoder(cols=['artists'], n_components=5, return_df=True, drop_invariant=True)
    df = encoder.fit_transform(songs['artists'], songs[y_name])

    # save the encoder model
    pickle.dump(encoder, open(path+"endcoder.sav", 'wb'))

    songs = df.join(songs)
    return songs

def cleanYear(songs):
    '''drop rows with invalid values or out of range'''
    songs = songs.dropna(subset=['year'])
    songs.loc[songs.year > 1900, 'year'] = 2020 - songs.year
    songs = songs.rename(columns={'year': 'yearsSinceCreation'})
    return songs

def mergeDuplicates(songs,y_name):
    '''merging groups of songs with the same name and artists taking mean values for the other columns'''
    songs = songs.drop_duplicates(subset=['artists', 'name'])
    songs.loc[:, 'explicit'] = round(songs.explicit)
    return songs

def removeEmpty(songs):
    for col in songs.columns:
        songs = songs.dropna(subset=[col])
    return songs

def dropCols(songs):
    songs = songs.drop(columns=['id', 'release_date'])
    return songs

def pre(songs):
    y_name = songs.columns[-1]
    
    if y_name == "popularity_level":
        path = "./preprocessing/models/classification/"
    else:
        path = "./preprocessing/models/regression/"
        
    print("preprossing start")
    songs = cleanYear(songs)
    print("remove empty data")
    songs = removeEmpty(songs)
    print("drop unUsed columns")
    songs = dropCols(songs)
    print("merge duplicates songs (might take a while) ")
    songs = mergeDuplicates(songs,y_name)
    print("clean artists (might take a while) ")
    songs = cleanArtists(songs, y_name, path)
    
    songs = songs.drop(columns=['name', 'artists'])
    songs.dropna(how='any',inplace=True)
        
    scaler = MinMaxScaler().fit(songs.iloc[:,0:-1])
    songs.iloc[:,0:-1] = scaler.transform(songs.iloc[:,0:-1])
    # save the encoder model
    pickle.dump(scaler, open(path+"scaler.sav", 'wb'))

    songsWithoutAritsts = songs.loc[:,songs.columns not in ['col_0', 'col_1', 'col_2', 'col_3', 'col_4']]
    print("preprossing end")
    
    return songsWithoutAritsts, songs

def preForNew(songs: DataFrame):
    y_name = songs.columns[-1]

    if y_name == "popularity_level":
        path = "./preprocessing/models/classification/"
    else:
        path = "./preprocessing/models/regression/"

    # year
    songs.loc[:, 'year'] = 2020 - songs.year
    songs = songs.rename(columns={'year': 'yearsSinceCreation'})

    # drop columns
    songs = songs.drop(columns=['id', 'release_date', 'name'])

    # artists
    encoder = pickle.load(open(path + "encoder.sav", 'rb'))
    df = encoder.transform(songs['artists'], songs[y_name])
    songs = df.join(songs)
    songs = songs.drop(columns=['artists'])
    
    # scaler
    scaler = pickle.load(open(path + "scaler.sav", 'rb'))
    songs.iloc[:,0:-1] = scaler.transform(songs.iloc[:,0:-1])

    songsWithoutAritsts = songs.loc[:,songs.columns not in ['col_0', 'col_1', 'col_2', 'col_3', 'col_4']]

    return songsWithoutAritsts, songs

def runPre():
    songsRegression = read_csv("../datasets/spotify_training.csv")
    songsClassification = read_csv("../datasets/spotify_training_classification.csv")

    songs, songsWithAritsts = pre(songsRegression)
    songs.to_csv("./dataSetCache/regression/songs.csv", index=False)
    songsWithAritsts.to_csv("./dataSetCache/regression/songsWithAritsts.csv", index=False)

    songs, songsWithAritsts = pre(songsClassification)
    songs.to_csv("./dataSetCache/classification/songs.csv", index=False)
    songsWithAritsts.to_csv("./dataSetCache/classification/songsWithAritsts.csv", index=False)
