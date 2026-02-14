from sklearn.tree import DecisionTreeClassifier

def train_decision_tree(X_train, y_train, X_test):

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model, X_test
