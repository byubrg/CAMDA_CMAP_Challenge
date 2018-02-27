from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn import preprocessing

#from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier #ensemble or neural_network?

#from sklearn.datatsets import make_classification

from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import sys, gzip

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

    X_train, X_test, y_train, y_test = train_test_split(features, answers, test_size=.30)

    scaler = StandardScaler()

    ## This sets the size of the scaler object
    scaler.fit(X_train)

    ## The MLP is super senesitive to feature scaling, so it is highly recommended to scale your data.
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    ## Build your classifier
    RF = RandomForestClassifier(max_depth=2, random_state=0)

    ## Place the training data in the MLP to train your algorithm
    RF.fit(X_train, y_train)

    ## This is where we test the trained algorithm
    predictions = RF.predict(X_test)
    y_prob = RF.predict_log_proba(X_test)

    ## This will show the confusion in a matrix that will tell how often we were correct 
    print(confusion_matrix(y_test,predictions))
    print(classification_report(y_test,predictions))
    for i in range(len(y_prob)) :
        print("Predicted value for item " + str(i + 1) + " : " + str(predictions[i]) + ", actual: " + str(y_test[i]))
        print("Probability : " + str(y_prob[i]))
    return RF


## This is the learner object that is trained on 70 percent of the data
learner = train(train_file)
