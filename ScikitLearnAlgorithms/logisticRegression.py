from sklearn import linear_model

# In our empirical analysis, the lbfgs solver performed better than the other four 
# alternative solvers. None of the other tested parameters significantly altered performance.
def logisticRegression(X_train, X_test, y_train) :
    classifier = linear_model.LogisticRegression(solver='lbfgs')

    ## Place the training data in the MLP to train your algorithm
    classifier.fit(X_train,y_train)
    ## This is where we test the trained algorithm
    predictions = classifier.predict(X_test)
    y_prob = classifier.predict_proba(X_test)

    return predictions, y_prob
    
