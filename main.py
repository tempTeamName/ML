from pandas import read_csv
from pre import pre
from reg import reg

songs = read_csv('clean.csv')
print(songs.describe())
# songs = pre(songs)
# songs.to_csv(path_or_buf="clean.csv", index=False)
degree = [1,2,3]
alpha = [0.1,0.3,0.5,0.7,1,10]
for degreeI in degree:
    for alphaI in alpha:
        print("degree :", degreeI, "\nalpha :",alphaI,"\n\n")
        reg(songs = songs,isTopFeatures= False, degree = degreeI , alpha = alphaI, normalize = True)
        print("\n\n==================================") 


# print("=============== top features ===============\n\n")
# print(" top, dg=3, a=0.5\n\n")
# reg(songs = songs,isTopFeatures= True, degree = 3, alpha = 0.5, normalize = False) 
# print("=============== all features ===============\n\n")
# print(" dg=3, a=0.5\n\n")
# reg(songs = songs,isTopFeatures= False, degree = 3, alpha = 0.5, normalize = False) 