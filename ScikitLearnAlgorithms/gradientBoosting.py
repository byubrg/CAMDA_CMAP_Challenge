## Look into this in the future
from sklearn.ensemble import GradientBoostingClassifier
def grad(X_train, X_test, y_train, rand='not_an_int') :
    grad = GradientBoostingClassifier(learning_rate = .31, max_depth = 3 )
    grad.fit(X_train,y_train)
    predictions = grad.predict(X_test)
    y_prob = grad.predict_proba(X_test)
    return predictions, y_prob

