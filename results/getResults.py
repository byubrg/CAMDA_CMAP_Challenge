import sys
from helper.calcuateAccuracy import *

ensembleLoc = sys.argv[1]
rfLoc = sys.argv[2]
logRegLoc = sys.argv[3]
actual = sys.argv[4]

def getResult(prediction, actual,results) :
    if int(actual) == 1 :
        if int(prediction) == int(actual) :
            results[0] += 1 
        else :
            results[1] += 1 
    else :
        if int(prediction) == int(actual) :
            results[2] += 1 
        else :
            results[3] += 1 
    return(results)

def compareFiles(predictions, actual) :
    with open(predictions, 'r') as f1 :
        with open(actual, 'r') as f2: 
            f1.readline()
            f2.readline()
            f2.readline()

            ## TP, FP, TN, FN
            MCF7Results = [0,0,0,0]
            PC3Results = [0,0,0,0]

            f1PredictionMCF7 = []
            f1PredictionPC3 = []

            for line in f1 :
                lineListf1 = line.strip('\n').split(',')
                f1PredictionMCF7.append(lineListf1[1])
                f1PredictionPC3.append(lineListf1[2])

            actual = []
            for line in f2 :
                lineListf2 = line.strip('\n').split(',')
                actual.append(lineListf2[2])
            
            if len(actual) != len(f1PredictionMCF7) :
                print("error")
                quit()
            if len(actual) != len(f1PredictionPC3) :
                print("error")
                quit()

            for i in range(len(actual)):
                MCF7Results = getResult(f1PredictionMCF7[i], actual[i], MCF7Results)
                PC3Results = getResult(f1PredictionPC3[i], actual[i], PC3Results)

            print("MCF7")
            print(MCF7Results)
            printConfusionCalculations(MCF7Results[0],MCF7Results[2],MCF7Results[1],MCF7Results[3])
            print("PC3")
            print(PC3Results)
            printConfusionCalculations(PC3Results[0],PC3Results[2],PC3Results[1],PC3Results[3])

print("ensemble")
compareFiles(ensembleLoc, actual)
print("RF")
compareFiles(rfLoc, actual)
print("logReg")
compareFiles(logRegLoc, actual)

