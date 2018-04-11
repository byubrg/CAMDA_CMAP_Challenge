from sklearn.ensemble import RandomForestClassifier #ensemble or neural_network?


def rf(X_train, X_test, y_train) :

    ## Build your classifier
    RF = RandomForestClassifier(n_estimators=25,
                                max_depth=9,
                                min_samples_split=2,
                                min_samples_leaf=1,
                                min_weight_fraction_leaf=0,
                                max_leaf_nodes=25,
                                bootstrap=False, 
                                random_state=randomSeed)

    ## Place the training data in the MLP to train your algorithm
    RF.fit(X_train, y_train)

    ## This is where we test the trained algorithm
    predictions = RF.predict(X_test)
    y_prob = RF.predict_log_proba(X_test)

    return predictions, y_prob

## This function illustrates how we optomized the parameters.
def rfo(X_train, X_test, y_train, optomization, randomSeed) :
    ## Build your classifier
    RF = RandomForestClassifier(n_estimators=25,
                                max_depth=9,
                                min_samples_split=2,
                                min_samples_leaf=1,
                                min_weight_fraction_leaf=0,
                                max_leaf_nodes=25,
                                bootstrap=False, 
                                random_state=randomSeed)

    ## Place the training data in the MLP to train your algorithm
    RF.fit(X_train, y_train)

    ## This is where we test the trained algorithm
    predictions = RF.predict(X_test)
    y_prob = RF.predict_log_proba(X_test)

    return predictions, y_prob
