from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X_train, y_train, X_test):

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, X_test
