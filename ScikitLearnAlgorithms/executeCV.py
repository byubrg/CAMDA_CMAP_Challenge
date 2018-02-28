from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
import numpy as np
import sys, gzip

#Import algorithm file
from mlp import mlp
from random_forest import rf

train_file = sys.argv[1]

def train(infile):
    with gzip.open(infile, 'r') as file :
        data = np.genfromtxt(file, delimiter='\t',dtype=str)

    ## Split the data up into features and answers
    answers = []        	
    features = []
    for row in data[1:,]:
        answers.append(row[1])
        features.append(row[2:])
	
    features = np.array(features)
    answers = np.array(answers) 

    y_test_final = np.array([])
    predictions_final = np.array([])
    y_prob_final = np.ndarray(shape=(0,2), dtype=int)

    ##first way to cross validate, not very easy to scale
#    scores = cross_val_score(mlp, features, answers, cv=10)
#    print(scores)

    ## Second way that is applicable, problem is that it just takes the first certain number. We don't know if there is a correlation
#    kf = KFold(n_splits=10)
#    for train, test in kf.split(features):

    ## Third way is the shuffle split.
    ss = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
    for train, test in ss.split(features):
        print("Training: %s \n Test: %s" % (train, test))	
        scaler = StandardScaler()
        X_train, X_test, y_train, y_test = features[train], features[test], answers[train], answers[test]
   
        ## This sets the size of the scaler object
        scaler.fit(X_train)

        ## The MLP is super senesitive to feature scaling, so it is highly recommended to scale your data.
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        ## This is where we can add the algorithms
#        predictions, y_prob = mlp(X_train, X_test, y_train)
        predictions, y_prob = rf(X_train, X_test, y_train)

        ## This will show the confusion in a matrix that will tell how often we were correct 
        y_test_final = np.concatenate([y_test_final,y_test])
        predictions_final = np.concatenate([predictions_final,predictions])
        y_prob_final = np.concatenate([y_prob_final,y_prob])

    print(confusion_matrix(y_test_final,predictions_final))
    print(classification_report(y_test_final,predictions_final))
    for i in range(len(y_prob_final)) :
        print("Predicted value for item " + str(i + 1) + " : " + str(predictions_final[i]) + ", actual: " + str(y_test_final[i]))
        print("Probability : " + str(y_prob_final[i]))


## This is the learner object that is trained on 70 percent of the data
learner = train(train_file)
