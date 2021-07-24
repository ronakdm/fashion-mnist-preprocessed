# fashion-mnist-preprocessed
Image classification datasets are commonly used for benchmarking ML models. However, this poses a challenge for algorithms that are not inherenetly suited to computer vision or convolutional neural network architectures. Thus, an alternative is to featurize images using pretrained neural networks. An issue with this approach is that representations by these networks such as ResNet50 can make the problem too easy to be descriptive. This repo contains code to download the Fashion MNIST dataset, featurize it using the local binary pattern method from computer vision, and make the problem a binary classification problem using the hardest to distinguish classes in this representation. This affords easy and quick benchmarking.

## Dependencies

The code requires Python 3 environment with the following packages.
```
- numpy
- torchvision
- scikit-learn
- scikit-image
- joblib
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


