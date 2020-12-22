from pandas import read_csv
from pre import pre
from reg import reg

songs = read_csv('spotify_training.csv')
songs = pre(songs)
print("=============== top features ===============")
reg(songs, True, 3, 0.5, False) 
print("=============== all features ===============")
reg(songs, False, 3, 0.5, False)
