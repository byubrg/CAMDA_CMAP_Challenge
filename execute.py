import numpy as np
import sys, gzip
import copy

## Import sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA

## Import customly adapted sklearn algorithm modules 
from ScikitLearnAlgorithms.mlp import * 
from ScikitLearnAlgorithms.randomForest import * 
from ScikitLearnAlgorithms.naiveBayes import * 
from ScikitLearnAlgorithms.kNearestNeighbor import * 
from ScikitLearnAlgorithms.svm import * 
from ScikitLearnAlgorithms.logisticRegression import * 
from ScikitLearnAlgorithms.gradientBoosting import grad

## Import Helper Modules
from helper.calcuateAccuracy import *
from helper.ensemble import ensemble
from helper.featureSelect import featureSelect

## Set File Locations
trainPC3 = sys.argv[1]
trainMCF7 = sys.argv[2]
testPC3 = sys.argv[3]
testMCF7 = sys.argv[4]
discretePredictionsOut = sys.argv[5]

def makePredictions(X_train,X_test,y_train) :
    scaler = StandardScaler()

    ## This sets the size of the scaler object
    scaler.fit(X_train)

    ## The MLP is super senesitive to feature scaling, so it is highly recommended to scale your data.
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    ####### This is where we want to implement the Ensemble method ###### 
#    predictions, y_prob = mlp(X_train, X_test, y_train)
#    predictions, y_prob = rf(X_train, X_test, y_train)
#    predictions, y_prob = naiveBayes(X_train, X_test, y_train)
#    predictions, y_prob = kNearestNeighbor(X_train, X_test, y_train)
#    predictions, y_prob = supportVM(X_train, X_test, y_train) ##Attention, this returns a y_prob of 0 because it doesn't work with the SVM
#    predictions, y_prob = logisticRegression(X_train, X_test, y_train)
#    predictions, y_prob = grad(X_train, X_test, y_train)
    predictions, y_prob = ensemble(X_train, X_test, y_train)

    return predictions, y_prob


def train(trainFile, selected = None):
    print(trainFile)
    with gzip.open(trainFile, 'r') as file :
        data = np.genfromtxt(file, delimiter='\t',dtype=str)

    ## Split the data up into features and answers
    answers = []        	
    features = []
    for row in data[1:,]:
        answers.append(row[1])
        features.append(row[2:])
	
    ## Convert to numpy arrays for algorithms
    features = np.array(features,dtype=float)
    answers = np.array(answers,dtype=float) 
    if selected != None:
        features = features[:,selected]
    ## Initialize prediction arrays
    y_test_final = np.array([])
    predictions_final = np.array([])
    y_prob_final = np.ndarray(shape=(0,2), dtype=int)

    ## We are using stradified fold cross validation.
    skf = StratifiedKFold(n_splits=10)
    i = 0

    ## Feature Selection needs to happen on each fold independently
    for train, test in skf.split(features, answers) :
        ## You can uncomment this row to see which indecis are used for the training and test sets for each fold
#        print("Training: %s \n Test: %s" % (train, test))	
        i += 1
        print("Fold:",i,sep=" ")
        X_train, X_test, y_train, y_test = features[train], features[test], answers[train], answers[test]

        ## this is a custom function that takes the top 25 % of the variance of the values 
        #X_train,X_test = featureSelect(X_train,X_test)
        #pca = PCA(n_components=.95)
        #pca.fit(X_train)
       
        #X_train = pca.transform(X_train)
        #X_test = pca.transform(X_test)

        predictions, y_prob = makePredictions(X_train,X_test,y_train) 

        ## This will show the confusion in a matrix that will tell how often we were correct 
        y_test_final = np.concatenate([y_test_final,y_test])
        predictions_final = np.concatenate([predictions_final,predictions])
        y_prob_final = np.concatenate([y_prob_final,y_prob])

    matrix = confusion_matrix(y_test_final,predictions_final),
    TP = matrix[0][1][1]
    FP = matrix[0][1][0]
    TN = matrix[0][0][0]
    FN = matrix[0][0][1]

    print("\nConfusion Matrix -",
          "   True Negative = zeros that were calculated correctly",
          "   False Negative = zeros that were calculated incorrectly",
          "   True Positive = ones that were calculated correctly",
          "   False Positive = ones that were calculated incorrectly",
          "\n[[True Negative,False Negative]",
          "[False Positive,True Positive]]\n",
          matrix[0],
          "\n",
          classification_report(y_test_final,predictions_final),
          sep='\n')
    printConfusionCalculations(TP, TN, FP, FN),

    ## You can uncomment this section to see which values are predicted incorrectly
#    for i in range(len(y_prob_final)) :
#        print("Predicted value for item " + str(i + 1) + " : " + str(predictions_final[i]) + ", actual: " + str(y_test_final[i]))
#        print("Probability : " + str(y_prob_final[i]))


def test(trainFile,testFile):
    with gzip.open(trainFile, 'r') as file :
        trainData = np.genfromtxt(file, delimiter='\t',dtype=str)
    with gzip.open(testFile, 'r') as file :
        testData = np.genfromtxt(file, delimiter='\t',dtype=str)

    ## training data
    y_train = []
    X_train = []
    for row in trainData[1:,]:
        y_train.append(row[1])
        X_train.append(row[2:])

    X_test = []
    for row in testData[1:,]:
        X_test.append(row[2:])

#    y_train = np.array(y_train,dtype=float)
    X_train = np.array(X_train,dtype=float) 
    X_test = np.array(X_test,dtype=float) 

    X_train,X_test = featureSelect(X_train,X_test)

    ## Convert to numpy arrays for algorithms
    X_test = np.array(X_test,dtype=float)
    X_train = np.array(X_train,dtype=float)

    predictions, y_prob = makePredictions(X_train,X_test,y_train) 

    return predictions, y_prob

def optomize(trainFile,cellLine,outFile,boolFeatureSelection,rangeOfParameterTested,rangeRandomSeed):
    with gzip.open(trainFile, 'r') as file :
        data = np.genfromtxt(file, delimiter='\t',dtype=str)

    ## Split the data up into features and answers
    answers = []        	
    features = []
    for row in data[1:,]:
        answers.append(row[1])
        features.append(row[2:])
	
    ## Convert to numpy arrays for algorithms
    features = np.array(features,dtype=float)
    answers = np.array(answers,dtype=float) 

    with open(outFile, 'w') as outFile :
        outFile.write("cellLine\trandomSeed\tparameterTested\tboolFeatureSelection\taccuracy\tsensitivity\tspecificity\tmcc\n")

        for parameterTested in rangeOfParameterTested :
            print("Parameter value:",parameterTested,sep="\t")
            for randomSeed in range(rangeRandomSeed) :
                ## Initialize prediction arrays
                y_test_final = np.array([])
                predictions_final = np.array([])
                y_prob_final = np.ndarray(shape=(0,2), dtype=int)

                ## We are using stradified fold cross validation.
                skf = StratifiedKFold(n_splits=10)
#                i = 0

                ## Feature Selection needs to happen on each fold independently
                for train, test in skf.split(features, answers) :
                    ## You can uncomment this row to see which indecis are used for the training and test sets for each fold
#                    print("Training: %s \n Test: %s" % (train, test))	
#                    i += 1
#                    print("Fold:",i,sep=" ")
                    X_train, X_test, y_train, y_test = features[train], features[test], answers[train], answers[test]

                    ## this is a custom function that takes the top 25 % of the variance of the values 
                    
                    if boolFeatureSelection == True :
                        X_train,X_test = featureSelect(X_train,X_test)
       
                    scaler = StandardScaler()

                    ## This sets the size of the scaler object
                    scaler.fit(X_train)

                    ## The MLP is super senesitive to feature scaling, so it is highly recommended to scale your data.
                    X_train = scaler.transform(X_train)
                    X_test = scaler.transform(X_test)

                    ## You will need to optomize your function
                    ## Change this to the function you are optomizing!!
                    predictions, y_prob = rfo(X_train, X_test, y_train, parameterTested, randomSeed)

                    ## This will show the confusion in a matrix that will tell how often we were correct 
                    y_test_final = np.concatenate([y_test_final,y_test])
                    predictions_final = np.concatenate([predictions_final,predictions])
                    y_prob_final = np.concatenate([y_prob_final,y_prob])

                matrix = confusion_matrix(y_test_final,predictions_final),
                TP = matrix[0][1][1]
                FP = matrix[0][1][0]
                TN = matrix[0][0][0]
                FN = matrix[0][0][1]

                accuracy, sensitivity, specificity, mcc = getConfusionInformation(TP, TN, FP, FN) 
               
                print(cellLine + "\t" + 
                              str(randomSeed) + "\t" +
                              str(parameterTested) + "\t" +
                              str(boolFeatureSelection) + "\t" +
                              str(accuracy) + "\t" +
                              str(sensitivity) + "\t" +
                              str(specificity) + "\t" +
                              str(mcc) + "\n")
                outFile.write(cellLine + "\t" + 
                              str(randomSeed) + "\t" +
                              str(parameterTested) + "\t" +
                              str(boolFeatureSelection) + "\t" +
                              str(accuracy) + "\t" +
                              str(sensitivity) + "\t" +
                              str(specificity) + "\t" +
                              str(mcc) + "\n")


            """
                print("iteration:",i,
                     "randomSeed:",randomSeed,
                     "numEstimators:",numEstimators,
                     "isFeatureSelectionImplemented",boolFeatureSelection,
                     "accuracy:",accuracy,
                     "sensitivity:",sensitivity,
                     "specificity:",specificity,
                     "mcc:",mcc,
                     sep = " ")
            """
def wrapper_function(trainFile):
    selected = []
    master_best_score = 0
    #this is the wrapper function it will continue to add the best feature to the selected column untill the accuracy stops increasing
    single = []
    
    with gzip.open(trainFile, 'r') as file :
        data = np.genfromtxt(file, delimiter='\t',dtype=str)

    ## Split the data up into features and answers
    answers = []        	
    features = []
    for row in data[1:,]:
        answers.append(row[1])
        features.append(row[2:])
	
    ## Convert to numpy arrays for algorithms
    features = np.array(features,dtype=float)
    answers = np.array(answers,dtype=float) 
    
    for i in range(0,len(features[0])):
        best_index = -1
        best_score = 0
        for j in range(0,len(features[0])):
            if j in selected:
                continue
            temp = copy.deepcopy(selected)
            temp.append(j)
            #sub set the feature to have only the column that are desired
            sub = features[:,temp]
            print(features[:,temp])
            print(len(sub[0]))
            total_correct = train(trainFile,temp)
            if total_correct is None:
                total_correct = 0
            if total_correct > best_score:
                best_score = total_correct
                best_index = j
        if best_score > master_best_score:
            master_best_score = best_score
            selected.append(best_index)
        else:
            break
    return selected


## Optimize, formating -> trainFile,outFile,boolFeatureSelection,valuesNumEstimators,rangeRandomSeed
"""
print("Training PC3\n")
valuesNumEstimators = list(range(101))
valuesNumEstimators = valuesNumEstimators[1:100:10]
#valuesNumEstimators = [True,False] 
print(valuesNumEstimators)
#optomize(trainPC3,"PC3","parameterOptomizationOutFile.txt",True,valuesNumEstimators,5)
#optomize(trainPC3,"PC3","parameterOptomizationOutFile.txt",False,valuesNumEstimators,5)
#optomize(trainMCF7,"MCF7","parameterOptomizationOutFile.txt",True,valuesNumEstimators,5)
#optomize(trainMCF7,"MCF7","parameterOptomizationOutFile.txt",False,valuesNumEstimators,5)
"""

#selected = wrapper_function(trainPC3)
selected = None
## TRAINING 
print("Training PC3\n")
train(trainPC3,selected)

print("\n\n\nTraining MCF7\n")
train(trainMCF7,selected)

"""
## TESTING
print("\n\n\nTesting PC3\n")
predictionsPC3, yprobPC3 = test(trainPC3,testPC3)

print("\n\n\nTesting MCF7\n")
predictionsMCF7, y_probMCF7 = test(trainMCF7,testMCF7)

with open(discretePredictionsOut, 'w') as dPO :
    dPO.write("Compound No. from Validation List,MCF7,PC3\n")
    for i in range(len(predictionsPC3)) :
        dPO.write(str(i+1) + "," + str(predictionsMCF7[i]) + "," + str(predictionsPC3[i]) + '\n')

"""
