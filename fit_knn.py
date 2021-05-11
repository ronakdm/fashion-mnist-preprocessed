print("Importing dependencies...")

import numpy as np
from utils import KNeighborsClassifierCV
import sys
import numpy as np
from sklearn.metrics import confusion_matrix
import pickle

# Cross-validation hyperparameters

max_n_neighbors = 10
n_splits = 5

if len(sys.argv) > 1:
    downsample_ratio = float(sys.argv[1])
else:
    downsample_ratio = 1

print("Loading training data...")

X_train = np.load("data/X_train.npy")
y_train = np.load("data/y_train.npy")

np.random.seed(123)
np.random.shuffle(X_train)
n = len(X_train)
idx = np.arange(n)[0 : int(n * downsample_ratio)]
X_train = X_train[idx, :]
y_train = y_train[idx]

print("Subsetting training data to %d/%d points..." % (int(n * downsample_ratio), n))

print(
    "Fitting adaptive k-neighest neighbors using %d-fold cross validation for k = 1, ..., %d..."
    % (n_splits, max_n_neighbors)
)

model = KNeighborsClassifierCV(
    max_n_neighbors=max_n_neighbors, n_splits=n_splits, verbose=True
).fit(X_train, y_train)

print("Loading test data...")

X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

if len(sys.argv) > 2:
    downsample_ratio = float(sys.argv[2])
else:
    downsample_ratio = 1
n = len(X_test)
idx = np.arange(n)[0 : int(n * downsample_ratio)]

print("Subsetting test data to %d/%d points..." % (int(n * downsample_ratio), n))

X_test = X_test[idx, :]
y_test = y_test[idx]

print("Computing evaluation metrics...")

best_params = model.get_params()
best_confusion_matrix = confusion_matrix(y_test, model.predict(X_test))

pickle.dump(best_params, open("params.p", "wb"))
with open("confusion_matrix.npy", "wb") as f:
    np.save(f, best_confusion_matrix)

# TODO: Most confusing pair of classes.
# TODO: binarizing data
# TODO: print shape

print("Done!")

