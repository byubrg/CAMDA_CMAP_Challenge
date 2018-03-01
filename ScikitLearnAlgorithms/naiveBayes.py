## This implements the Gaussian Naive Bayes algorithm. If time find out what the other algorithms are.
from sklearn.naive_bayes import GaussianNB

def naiveBayes(X_train, X_test, y_train) :
    gnb = GaussianNB()

    ## Place the training data in the MLP to train your algorithm
    gnb.fit(X_train,y_train)

    ## This is where we test the trained algorithm
    predictions = gnb.predict(X_test)
    y_prob = gnb.predict_proba(X_test)

    return predictions, y_prob
