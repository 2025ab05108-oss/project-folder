from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def train_logistic(X_train, y_train, X_test):

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model, X_test
