import numpy as np
from utils import KNeighborsClassifierCV
from sklearn.metrics import confusion_matrix

# Cross-validation hyperparameters

max_n_neighbors = 10
n_splits = 5

X_train = np.load("data/X_train.npy")
y_train = np.load("data/y_train.npy")
X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

print(
    "Fitting adaptive k-neighest neighbors using %d-fold cross validation for k = 1, ..., %d..."
    % (n_splits, max_n_neighbors)
)

model = KNeighborsClassifierCV(max_n_neighbors=max_n_neighbors, n_splits=n_splits).fit(
    X_train, y_train
)

print(model.get_params())

best_confusion_matrix = confusion_matrix(y_test, model.predict(X_test))

with open("confusion_matrix.npy", "wb") as f:
    np.save(f, y_test)

# TODO: Most confusing pair of classes.
# TODO: binarizing data
# TODO: print shape

