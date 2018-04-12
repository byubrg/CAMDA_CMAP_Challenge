# Importing algorithms from Scikit Learn (This is stuff I added)
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier

def ensemble(X_train, X_test, y_train):
    alg1 = linear_model.LogisticRegression()
    alg2 = svm.SVC(probability=True)
    alg3 = GaussianNB()
    alg4 = KNeighborsClassifier(n_neighbors=5) 
    alg5 = MLPClassifier(hidden_layer_sizes=(30,30,30))
    alg6 = GradientBoostingClassifier()        
    estimators = []

    estimators.append(('logistic', alg1))
    estimators.append(('svm', alg2))
    estimators.append(('Gussian', alg3))
    estimators.append(('KNeighbors', alg4))
    estimators.append(('MLP', alg5))
    estimators.append(('grad',alg6))   
    ensemble = VotingClassifier(estimators, voting='soft', weights=[1,1,2,2,2,2])
    
    ensemble.fit(X_train, y_train)

    predictions = ensemble.predict(X_test)
    y_prob = ensemble.predict_proba(X_test)

    return predictions, y_prob
