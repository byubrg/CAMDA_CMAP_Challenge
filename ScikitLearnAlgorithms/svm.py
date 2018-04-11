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



## Many of the support vector machine parameters are based on the kernel that is set. The default, 'rbf' kernel, does not have
## very many parameters associated with it. 'rbf' has the highest accuracy out of all the other kernels such as 'linear' and
## 'poly'. Therefore, no changes have been made to the support vector machine int terms of parameter optimization.

