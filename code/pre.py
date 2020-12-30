import pandas as pd 
import numpy as np 
import math
from random import randrange
import random
import string
import category_encoders as ce 
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer, minmax_scale
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

# drop rows with invalid values and destringify the list of artists 
def cleanArtists(songs, y_name):
    songs['artists'] = songs['artists'].apply(lambda x: x[1:-1].split(', ') if(type(x) == str and len(x)) else [])
    songs['artists'] = songs['artists'].apply(lambda x: list(map(lambda y: y[1:-1], x)) )
    encoder = ce.HashingEncoder(cols=['artists'], n_components=5, return_df=True, drop_invariant=True)
    df = encoder.fit_transform(songs['artists'], songs[y_name])
    songs = df.join(songs)
    return songs

# drop rows with invalid values or out of range (797 rows)
def cleanYear(songs):
     songs = songs.dropna(subset=['year'])
     songs.loc[songs.year > 1900, 'year'] = 2020 - songs.year
     songs = songs.rename(columns={'year': 'yearsSinceCreation'})
     return songs

# merging groups of songs with the same name and artists taking mean values for the other columns
def mergeDuplicates(songs,y_name):
    songs = songs.groupby(['artists', 'name'], as_index=False).agg({
        'valence':np.average,
        'yearsSinceCreation':np.average,
        'acousticness':np.average,
        'danceability':np.average,
        'duration_ms':np.average,
        'energy':np.average,
        'instrumentalness':np.average,
        'liveness':np.average,
        'loudness':np.average,
        'tempo':np.average,
        'speechiness':np.average,
        'explicit':np.average,
        'mode':np.average,
        y_name:np.average
    })
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
    print("preprossing start")
    songs = cleanYear(songs)
    print("remove empty data")
    songs = removeEmpty(songs)
    print("drop unUsed columns")
    songs = dropCols(songs)
    # print("merge duplicates songs (might take a while) ")
    # songs = mergeDuplicates(songs)
    print("clean artists (might take a while) ")
    songsWithArtists = cleanArtists(songs, y_name)
    
    songs = songs.drop(columns=['name', 'artists'])
    songs.dropna(how='any',inplace=True)
    
    songs.iloc[:,0:-1] = minmax_scale(songs.iloc[:,0:-1])
    
    songsWithArtists = songsWithArtists.drop(columns=['name', 'artists'])
    songsWithArtists.dropna(how='any',inplace=True)

    songsWithArtists.iloc[:,0:-1] = minmax_scale(songsWithArtists.iloc[:,0:-1])
    print("preprossing end")
    
    return songs, songsWithArtists