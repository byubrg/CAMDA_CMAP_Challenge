from math import sqrt

def getAccuracy(truePositives, trueNegatives, falsePostives, falseNegatives):
    return((truePositives + trueNegatives) / float(truePositives + trueNegatives + falseNegatives + falsePostives))

def getSensitivity(truePositives, falseNegatives):
    return(truePositives/float(truePositives + falseNegatives))

def getSpecificity(trueNegatives, falsePostives):
    return(trueNegatives/float(trueNegatives + falsePostives))

def getMCC(truePositives, trueNegatives, falsePostives, falseNegatives):
    return((truePositives * trueNegatives - falsePostives * falseNegatives) / sqrt((truePositives + falsePostives) * (truePositives + falseNegatives) * (trueNegatives + falsePostives) * (trueNegatives + falseNegatives)))

def printConfusionCalculations(TP, TN, FP, FN) : 
    print("accuracy: " + str(getAccuracy(TP, TN, FP, FN)))
    print("sensitivity: " + str(getSensitivity(TP, FN)))
    print("specificity: " + str(getSpecificity(TN, FP)))
    print("MCC: " + str(getMCC(TP, TN, FP, FN)))

def getConfusionInformation(TP, TN, FP, FN) :
    return getAccuracy(TP, TN, FP, FN), getSensitivity(TP, FN), getSpecificity(TN, FP), getMCC(TP, TN, FP, FN)
