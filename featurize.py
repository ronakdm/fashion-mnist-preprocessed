import numpy as np
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from skimage.feature import local_binary_pattern

# Local binary pattern hyperparameters

radius = 3
n_points = 8 * radius
method = "default"

# Download data

print("Downloading Fashion MNIST dataset...")

training_data = FashionMNIST(
    root="data", train=True, download=True, transform=ToTensor()
)
test_data = FashionMNIST(root="data", train=False, download=True, transform=ToTensor())

# Featurize

print("Featurizing using local binary patterns...")

train_np = training_data.data
test_np = test_data.data
n_train = len(train_np)
n_test = len(test_np)

n_width, n_height = train_np[0].shape

X_train = np.zeros((n_train, n_width * n_height))
y_train = np.zeros(n_train)
X_test = np.zeros((n_test, n_width * n_height))
y_test = np.zeros(n_test)

for i in range(n_train):

    if i % 1000 == 0:
        print("Processing training image %d/%d..." % (i, n_train))

    image = train_np[i]
    label = int(training_data[i][1])

    X_train[i, :] = local_binary_pattern(image, n_points, radius, method).flatten()
    y_train[i] = label

for i in range(n_test):

    if i % 1000 == 0:
        print("Processing test image %d/%d..." % (i, n_test))

    image = test_np[i]
    label = int(test_data[i][1])

    X_test[i, :] = local_binary_pattern(image, n_points, radius, method).flatten()
    y_test[i] = label

# Save features and labels

print("Saving features and original labels...")

with open("data/X_train.npy", "wb") as f:
    np.save(f, X_train)
with open("data/y_train.npy", "wb") as f:
    np.save(f, y_train)
with open("data/X_test.npy", "wb") as f:
    np.save(f, X_test)
with open("data/y_test.npy", "wb") as f:
    np.save(f, y_test)
