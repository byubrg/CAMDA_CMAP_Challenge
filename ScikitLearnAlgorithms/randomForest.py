from sklearn.ensemble import RandomForestClassifier #ensemble or neural_network?


def rf(X_train, X_test, y_train) :

    ## Build your classifier
    RF = RandomForestClassifier(n_estimators=100, random_state=0)

    ## Place the training data in the MLP to train your algorithm
    RF.fit(X_train, y_train)

    ## This is where we test the trained algorithm
    predictions = RF.predict(X_test)
    y_prob = RF.predict_log_proba(X_test)

    return predictions, y_prob

def rfo(X_train, X_test, y_train, numEstimators, randomSeed) :
    ## Build your classifier
    RF = RandomForestClassifier(n_estimators=numEstimators, random_state=randomSeed)

    ## Place the training data in the MLP to train your algorithm
    RF.fit(X_train, y_train)

    ## This is where we test the trained algorithm
    predictions = RF.predict(X_test)
    y_prob = RF.predict_log_proba(X_test)

    return predictions, y_prob
