import numpy as np

# confusion_matrix = np.load("confusion_matrix.npy")
confusion_matrix = np.random.normal(size=(10, 10))
n_classes = len(confusion_matrix)

# Identify most confusion classes.

argmax = None
maximum = -np.inf
for i in range(n_classes):
    for j in range(n_classes):
        if confusion_matrix[i, j] > maximum:
            maximum = confusion_matrix[i, j]
            argmax = (i, j)

# Subset the data to include only those from the given classes.

i, j = argmax
for split in ["train", "test"]:
    X = np.load("data/X_%s.npy" % split)
    y = np.load("data/y_%s.npy" % split)
    idx = np.logical_or(y == i, y == j)

    with open("data/X_bin_%s.npy" % split, "wb") as f:
        np.save(f, X[idx])
    with open("data/y_bin_%s.npy" % split, "wb") as f:
        np.save(f, y[idx])

