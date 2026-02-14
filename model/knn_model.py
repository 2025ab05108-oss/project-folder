from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

def train_knn(X_train, y_train, X_test):

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    return model, X_test
