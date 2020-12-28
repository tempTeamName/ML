from pandas import DataFrame, read_csv

df = read_csv('spotify_training_classification.csv')
print(df[:].describe())