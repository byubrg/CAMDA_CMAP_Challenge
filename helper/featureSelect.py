import numpy as np

def featureSelect(train_features, test_features):

    ## Concatenates the two arrays to get the combined array
    combined = np.concatenate((train_features, test_features),axis=0)

    ## This takes the variance of the 2d np array on the zero axis.
    ## For example indeci one of the first array will be compared with the first 
    ## indecies of the other arrays to calculate the variance the variance of these 
    ## will be in the first indeci of the resulting 1d array..
    combinedVariance = np.var(combined, axis=0)

    numFeatures = len(combinedVariance)

    ## Gets the the value of 1/4 the number of features 
    oneFourth = int(numFeatures * 5/ 100)

    ## This grabs the top 25 variance indecis
    wantedIndecis = np.argpartition(combinedVariance,-oneFourth)[-oneFourth:]

    print(len(wantedIndecis))
    return train_features[:,wantedIndecis], test_features[:,wantedIndecis]

