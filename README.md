# fashion-mnist-preprocessed
Code to download, featurize, and binarize the Fashion MNIST dataset for easy benchmarking.

## Dependencies

The code requires Python 3 environment with the following packages.
```
- torchvision
- scikit-image
- numpy
```

## Instructions

Create a `data` folder in the root directory of the repo. Run the following to download the Fashion MNIST dataset and save the features and original labels.
```
python featurize.py
```
After that, run the following to use adaptive k-nearest neighbors to find the hardest pair of classes to distinguish. The examples from this class becomes a subset for a binary classification problem.
```
python binarize.py
```


