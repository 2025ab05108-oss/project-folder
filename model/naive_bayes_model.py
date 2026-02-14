from sklearn.naive_bayes import GaussianNB

def train_naive_bayes(X_train, y_train, X_test):

    model = GaussianNB()
    model.fit(X_train, y_train)

    return model, X_test
