# Importing algorithms from Scikit Learn (This is stuff I added)
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

def ensemble(X_train, X_test, y_train):
    svmObject = svm.SVC(probability=True)
#    gnb = GaussianNB()
    MLP = MLPClassifier(hidden_layer_sizes=(30,30,30,30,30,30,30,30,30,30), learning_rate_init = .0376)
    neigh = KNeighborsClassifier(n_neighbors=8, weights='distance')
    logReg = linear_model.LogisticRegression(solver='lbfgs')
    grad = GradientBoostingClassifier(learning_rate = .31, max_depth = 3 )
    RF = RandomForestClassifier(n_estimators=25,
                                max_depth=9,
                                min_samples_split=2,
                                min_samples_leaf=1,
                                min_weight_fraction_leaf=0,
                                max_leaf_nodes=25,
                                bootstrap=False,
                                random_state=0)
        
    estimators = []

    estimators.append(('logistic', logReg))
    estimators.append(('svm', svmObject))
#    estimators.append(('Gussian', gnb))
    estimators.append(('KNeighbors', neigh))
    estimators.append(('MLP', MLP))
    estimators.append(('RF', RF))
    estimators.append(('grad', grad))

    ensemble = VotingClassifier(estimators, voting='soft')
    
    ensemble.fit(X_train, y_train)

    predictions = ensemble.predict(X_test)
    y_prob = ensemble.predict_proba(X_test)

    return predictions, y_prob
