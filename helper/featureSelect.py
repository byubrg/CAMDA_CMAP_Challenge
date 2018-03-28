import numpy as np

def featureSelect(features):

    ## find the variance of every column
    variance = []
    for columnNum  in range(len(features[0])):
    	colData = []
    	for position in range(len(features)):
    		colData.append(float(features[position][columnNum]))
    	variance.append(np.var(colData))
    # find the top 25%
    varianceSorted = sorted(variance)
    cutOffValue = varianceSorted[3 * int(len(varianceSorted)/4)]
    top25 = []
    top25values = []
    for rowNum in range(len(features)):
        myrow = []
        for index in range(len(features[0])):
    	       if variance[index] >= cutOffValue:
    		             myrow.append(float(features[rowNum][index]))
        top25values.append(myrow)
    return np.array(top25values)
