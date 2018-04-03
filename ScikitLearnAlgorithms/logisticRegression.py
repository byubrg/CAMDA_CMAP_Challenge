from sklearn import linear_model

def logisticRegression(X_train, X_test, y_train) :
    classifier = linear_model.LogisticRegression()

    ## Place the training data in the MLP to train your algorithm
    classifier.fit(X_train,y_train)
    ## This is where we test the trained algorithm
    predictions = classifier.predict(X_test)
    y_prob = classifier.predict_proba(X_test)

    return predictions, y_prob
