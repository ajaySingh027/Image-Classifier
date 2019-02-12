# Image-Classifier
Implemented a convolutional neural network to classify images from the CIFAR-10 dataset.


## Description
In this project, I trained an image classifier to recognize different species of flowers. 
Imagine using something like this in a phone app that tells us the name of the flower our camera is looking at. 
In practice I trained this classifier, then exported it for use in our application. 
And used this dataset of 102 flower categories.


The project is broken down into multiple steps:
- Load and preprocess the image dataset
- Train the image classifier on your dataset
- Use the trained classifier to predict image content


## Prerequisites
The Code is written in Python 3. If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip.

To install pip run in the command Line
```python -m ensurepip -- default-pip```

to upgrade it
```python -m pip install -- upgrade pip setuptools wheel```

to upgrade Python
```pip install python -- upgrade```

Additional Packages that are required are: [Numpy](http://www.numpy.org/), [Pandas](https://pandas.pydata.org/), [MatplotLib](https://matplotlib.org/), [Pytorch](https://pytorch.org/).
You can donwload them using pip
```pip install numpy pandas matplotlib```

or [conda](https://anaconda.org/anaconda/python)
```conda install numpy pandas matplotlib pil```

In order to intall Pytorch head over to the Pytorch site select your specs and follow the instructions given.
