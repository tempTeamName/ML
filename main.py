from pandas import read_csv
from pre import pre
from reg import reg

def test(songs):
    degree = [1,2,3,4]
    alpha = [0.01,0.1,0.5,1,10]
    for degreeI in degree:
        for alphaI in alpha:
            print("degree :", degreeI, "\nalpha :",alphaI,"\n\n")
            reg(songs = songs,isTopFeatures= False, degree = degreeI , alpha = alphaI, normalize = True)
            print("\n\n==================================") 



if __name__ == "__main__":
    # songs = read_csv("spotify_training.csv")
    # songs = pre(songs, False)
    # songs.to_csv('clean.csv')
    
    songs = read_csv("clean.csv")
    print("all selected freatures with degree 3 and anlpha 0.01")
    reg(songs = songs,isTopFeatures= False, degree = 3 , alpha = 0.01, normalize = True)
    print("all selected freatures with degree 4 and anlpha 0.01 (current best fit) ")
    reg(songs = songs,isTopFeatures= False, degree = 4 , alpha = 0.01, normalize = True)
    print("top corr freatures with degree 3 and anlpha 0.01")
    reg(songs = songs,isTopFeatures= True, degree = 3 , alpha = 0.01, normalize = True)
    print("top corr freatures with degree 4 and anlpha 0.01")
    reg(songs = songs,isTopFeatures= True, degree = 4 , alpha = 0.01, normalize = True)

# best res with  dg = 4 , alpha : 0.01, all selected features