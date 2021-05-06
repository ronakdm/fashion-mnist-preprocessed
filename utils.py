import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.exceptions import NotFittedError
from joblib import Parallel, delayed


class KNeighborsClassifierCV:
    def __init__(self, max_n_neighbors=10, n_splits=5, score=accuracy_score):
        self.max_n_neighbors = max_n_neighbors
        self.n_splits = n_splits
        self.score = score

    def fit(self, X, y):

        kf = KFold(n_splits=self.n_splits)
        scores = np.zeros((self.max_n_neighbors, self.n_splits))

        for i, (train_index, val_index) in enumerate(kf.split(X)):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            def evaluate(k):
                model = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
                return self.score(y_val, model.predict(X_val))

            scores[:, i] = Parallel(n_jobs=-2)(
                delayed(evaluate)(k) for k in range(1, self.max_n_neighbors + 1)
            )

        best_n_neighbors = np.argmax(scores.mean(axis=1))
        self.best_model = KNeighborsClassifier(n_neighbors=best_n_neighbors).fit(X, y)

        return self.best_model

    def predict(self, X):
        if not hasattr(self, "best_model"):
            raise NotFittedError(
                "This KNeighborsClassifierCV instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
            )

        return self.best_model.predict(X)

    def get_params(self):
        if not hasattr(self, "best_model"):
            raise NotFittedError(
                "This KNeighborsClassifierCV instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
            )

        return self.best_model.get_params()
