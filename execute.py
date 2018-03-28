import numpy as np
import sys, gzip

## Import sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold

#Import customly adapted sklearn algorithm modules 
from ScikitLearnAlgorithms.mlp import mlp 
from ScikitLearnAlgorithms.randomForest import rf
from ScikitLearnAlgorithms.naiveBayes import naiveBayes
from ScikitLearnAlgorithms.kNearestNeighbor import kNearestNeighbor
from ScikitLearnAlgorithms.svm import supportVM 
from ScikitLearnAlgorithms.logisticRegression import logisticRegression

#Import Helper Modules
from helper.calcuateAccuracy import *
from helper.ensemble import ensemble
from helper.featureSelect import featureSelect

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
    predictions, y_prob = mlp(X_train, X_test, y_train)
#    predictions, y_prob = rf(X_train, X_test, y_train)
#    predictions, y_prob = naiveBayes(X_train, X_test, y_train)
#    predictions, y_prob = kNearestNeighbor(X_train, X_test, y_train)
#    predictions, y_prob = supportVM(X_train, X_test, y_train) ##Attention, this returns a y_prob of 0 because it doesn't work with the SVM
#    predictions, y_prob = logisticRegression(X_train, X_test, y_train)
#    predictions, y_prob = ensemble(X_train, X_test, y_train)

    return predictions, y_prob


def train(trainFile):
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

    ## Initialize prediction arrays
    y_test_final = np.array([])
    predictions_final = np.array([])
    y_prob_final = np.ndarray(shape=(0,2), dtype=int)

    ## We are using stradified fold cross validation.
    skf = StratifiedKFold(n_splits=10)
    i = 0

    ## Feature Selection needs to happen on each fold independently
#    features = featureSelect(features)
    for train, test in skf.split(features, answers) :
        ## You can uncomment this row to see which indecis are used for the training and test sets for each fold
#        print("Training: %s \n Test: %s" % (train, test))	
        i += 1
        print("Fold:",i,sep=" ")
        X_train, X_test, y_train, y_test = features[train], features[test], answers[train], answers[test]

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

#    X_train = featureSelect(X_train)

    ## Convert to numpy arrays for algorithms
    X_test = np.array(X_test,dtype=float)
    X_train = np.array(X_train,dtype=float)

    predictions, y_prob = makePredictions(X_train,X_test,y_train) 

    return predictions, y_prob



## This is the learner object that is trained on 70 percent of the data
print("Training PC3\n")
#train(trainPC3)

print("\n\n\nTraining MCF7\n")
#train(trainMCF7)

## TESTING
print("\n\n\nTesting PC3\n")
predictionsPC3, yprobPC3 = test(trainPC3,testPC3)

print("\n\n\nTesting MCF7\n")
predictionsMCF7, y_probMCF7 = test(trainMCF7,testMCF7)

with open(discretePredictionsOut, 'w') as dPO :
    dPO.write("Compound No. from Validation List,MCF7,PC3\n")
    for i in range(len(predictionsPC3)) :
        dPO.write(str(i+1) + "," + str(predictionsMCF7[i]) + "," + str(predictionsPC3[i]) + '\n')
