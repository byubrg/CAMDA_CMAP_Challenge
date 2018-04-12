from sklearn.neural_network import MLPClassifier

def mlp(X_train, X_test, y_train, nodes=(30,30,30), rand='not_an_int') :
    MLP = MLPClassifier(hidden_layer_sizes=(30,30,30,30,30,30,30,30,30,30), learning_rate_init = .0376) 

    ## Place the training data in the MLP to train your algorithm
    MLP.fit(X_train,y_train)

    ## This is where we test the trained algorithm
    predictions = MLP.predict(X_test)
    y_prob = MLP.predict_proba(X_test)

    return predictions, y_prob
