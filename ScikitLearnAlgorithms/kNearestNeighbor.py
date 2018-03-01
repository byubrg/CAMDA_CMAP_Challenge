## This implements the Gaussian Naive Bayes algorithm. If time find out what the other algorithms are.
from sklearn.neighbors import KNeighborsClassifier

def kNearestNeighbor(X_train, X_test, y_train) :
    neigh = KNeighborsClassifier(n_neighbors=5)

    ## Place the training data in the MLP to train your algorithm
    neigh.fit(X_train,y_train)

    ## This is where we test the trained algorithm
    predictions = neigh.predict(X_test)
    y_prob = neigh.predict_proba(X_test)

    return predictions, y_prob

