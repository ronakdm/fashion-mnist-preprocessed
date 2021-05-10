import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.exceptions import NotFittedError
from joblib import Parallel, delayed
import time
import datetime

def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


class KNeighborsClassifierCV:
    def __init__(self, max_n_neighbors=10, n_splits=5, score=accuracy_score, verbose=False):
        self.max_n_neighbors = max_n_neighbors
        self.n_splits = n_splits
        self.score = score
        self.verbose = verbose

    def fit(self, X, y):

        kf = KFold(n_splits=self.n_splits)
        scores = np.zeros((self.max_n_neighbors, self.n_splits))

        for i, (train_index, val_index) in enumerate(kf.split(X)):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            fold_tic = time.time()

            def evaluate(k):

                tic = time.time()
                model = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
                toc = time.time()

                if self.verbose:
                    print("Fold %d/%d:  %d-nearest neighbors fit time: %s." % (i + 1, self.n_splits, k, format_time(toc-tic)))

                return self.score(y_val, model.predict(X_val))

            scores[:, i] = Parallel(n_jobs=-2)(
                delayed(evaluate)(k) for k in range(1, self.max_n_neighbors + 1)
            )

            fold_toc = time.time()
            
            if self.verbose:
                print("Fold %d/%d time: %s." % (i + 1, self.n_splits, format_time(fold_toc-fold_tic)))

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