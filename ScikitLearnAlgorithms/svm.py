from sklearn import svm

def supportVM(X_train, X_test, y_train) :
#    classifier = svm.SVC(probability=True) ## Look into this if time 
    classifier = svm.SVC() 

    ## Place the training data in the MLP to train your algorithm
    classifier.fit(X_train,y_train)

    ## This is where we test the trained algorithm
    predictions = classifier.predict(X_test)

#    y_prob = classifier.predict_proba(X_test) ## We can't run this function for the svm. It doesn't work unless we set the probability to True
    y_prob =  np.ndarray(shape=(0,2), dtype=int)

    return predictions, y_prob
