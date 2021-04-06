
# Deep Learning for Visual Computing - Assignment 1

The first assignment allows you to become familiar with basic dataset handling, image processing, and machine learning.

This text or the reference code might not be without errors. If you find a significant error (not just typos but errors that affect the assignment), please contact us via [email](mailto:dlvc@cvl.tuwien.ac.at). Students who find and report such errors will get extra points.

## Part 0

This part is about setting up the environment for developing. 

All assignments will be implemented in Python 3 and [PyTorch](https://pytorch.org/). So first make sure Python 3.6 or newer is installed on your computer as that's the minimal requirement of the most recent PyTorch version. If not, [download](https://www.python.org/downloads/) and install a recent version.

Then setup, create, and enable a [virtualenv](https://virtualenv.pypa.io/en/stable/). This facilitates package installation and ensures that these packages don't interfere with other Python code you might already have. Once done, make sure `$ python --version` returns something like `python 3.7.0`. Finally, install the core packages we'll need:

    pip install numpy opencv-python

The PyTorch setup varies a bit depending on the OS, see [here](https://pytorch.org/). Use a version with CUDA only if you have an Nvidia GPU. In any case, ensure to install the current version of PyTorch, which is 1.8.0. This is the version we will use for testing all assignments and if they fail due to version issues, you'll get significant point deductions. Confirm this via:

    python -c "import torch; print(torch.__version__)"

## Part 1

Download the reference code from [here](https://smithers.cvl.tuwien.ac.at/theitzinger/dlvc_ss21_public/-/tree/master/assignments/reference), making sure that the file structure is preserved, and rename the root folder to something other than `reference`. Read the code and make sure that you understand what the individual classes and methods are doing.

[Download](https://www.cs.toronto.edu/~kriz/cifar.html) and extract the *Python* version of the CIFAR10 dataset somewhere *outside* the code folder. Read the website to understand which classes there are and how the data are structured and read.

Then implement the `PetsDataset` (`datasets/pets.py`). Some parts of this assignment will be tested automatically so make sure to follow the instructions in the code exactly (note, for example the specified data types). Make sure the following applies. If not, you made a mistake somewhere:

* Number of samples in the individual datasets: 7959 (training), 2041 (validation), 2000 (test).
* Total number of cat and dog samples: 6000 per class
* Image shape: always `(32, 32, 3`, image type: always `uint8`
* Labels of first 10 training samples: `0 0 0 0 1 0 0 0 0 1`
* Make sure that the color channels are in BGR order (not RGB) by displaying the images and verifying the colors are correct (`cv2.imshow`, `cv2.imwrite`).

Do not change any other files and do not create additional files.

## Part 2

Make sure you have the most recent [reference code](https://smithers.cvl.tuwien.ac.at/theitzinger/dlvc_ss21_public/-/tree/master/assignments/reference). Next to updated code we will occasionally add clarifications or correct errors. You can check the individual commits to see what has changed.

In this part we will implement common functionality for classifier training. As we'll see in the lecture, training and testing is almost always done in mini-batches, with each being a small part of the whole data. To do so, finish the `BatchGenerator` class in `batches.py`. Make sure to read the comments and implement type and value checks accordingly.

The `BatchGenerator`'s constructor has an optional `op` argument that is a function. If this argument is given, the generator will apply this function to the data of every sample before adding it to a batch. This is a flexible mechanism that will later allow us to implement data augmentation. For now we'll use it to transform the data to the form expected by classifiers such as KNN. For this we need to convert the images to float vectors, as covered in the lecture. To do so, implement the `type_cast`, `vectorize`, `add` and `mul` functions inside `ops.py`. These are functions that return other functions. See the `chain` function, which is already implemented for reference. That function allows for chaining other operations together like so:

```python
op = ops.chain([
    ops.vectorize(),
    ops.type_cast(np.float32),
    ops.add(-127.5),
    ops.mul(1/127.5),
])
```

We will use the `add()` and `mul()` operations for basic input normalization. The above arguments will scale the vector entries to the interval `[-1, 1]`.

To test the batch generator make sure the following applies (the following list assumes that the above op chain is used):

* The number of training batches is `1` if the batch size is set to the number of samples in the dataset
* The number of training batches is `16` if the batch size is set to 500
* The data and label shapes are `(500, 3072)` and `(500,)`, respectively, unless for the last batch which will be smaller
* The data type is always `np.float32` and the label type is integral (for example one of the `np.int` and `np.uint` variants)
* The first sample of the first training batch returned *without shuffling* has label `0` and data `[-0.09019608 -0.01960784 -0.01960784 -0.28627452 -0.20784315 ...]`.
* The first sample of the first training batch returned *with shuffling* must be random.

Finally we will use accuracy as the performance measure for our classifiers. See the lecture slides for how this measure is defined and implement the `Accuracy` class in `test.py` accordingly. This class supports batch-wise updates which will be handy in the future (we already talked about minibatches in the lecture).

## Part 3

In this part we will implement a simple classifier and test all of the code implemented so far. At this point the classifier will not be a deep neural network. Instead you can choose a more "traditional" type from the `scikit-learn` package. Finish the `SimpleClassifier` in `simple.py` (you can change the class name to better fit your choice of classifier but don't change the filename). The goal is not to find the best classifier possible but rather to implement a simple classifier that allows us to obtained baseline results which we can compare future models against. Good choices are types that were already covered in the lecture such as:

1. Nearest Neighbor - see `sklearn.neighbors.KNeighborsClassifier`
2. Linear classifiers - see `sklearn.linear_model.SGDClassifier`. Here the argument `loss='log'` gives a logistic regression classifier which is equivalent to a linear classifier with softmax normalization and cross entropy loss. Note that `sklearn` also directly implements `sklearn.linear_model.LogisticRegression` which works as well but gives you less parameters to tune (see requirements for `simple_cats_dogs.py` below)

Feel free to try other types of classifiers but stick to "traditional" methods - so stay away from `sklearn.neural_network`. For some bonus points you can implement more than one type.

Finally, combine the functionality implemented so far in a script `simple_cats_dogs.py` that does the following, in this order:

1. Load the training, validation, and test sets as individual `PetsDataset`s.
2. Create a `BatchGenerator` for each one. Traditional classifiers don't usually train in batches so you can set the minibatch size equal to the number of dataset samples to get a single large batch - unless you choose a classifier that does require multiple batches.
3. Implement random or grid search (your choice) to tune one ore more hyperparameter values (such as `k` for KNN classifiers). Test at least 10 values. This is not a lot but depending on your choice of classifier and parameters can take a long time.
4. For each parameter to test, "train" a `SimpleClassifier` and then calculate the accuracy on the validation set.
5. Report the best parameters found and the corresponding validation accuracy.
6. Compute and report the accuracy on the test set with these parameters.

Steps 3 to 6 must be implemented in a generic way using a loop, like so:

```python
for k in range(1, 101, 10):  # grid search example
    knn = SimpleClassifier(..., k) # KNN classifier example
    accuracy = Accuracy()
    # train and compute validation accuracy ...
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

knn = SimpleClassifier(..., best_k)
# compute test accuracy
```

## Report

Write a short report (2 to 3 pages) that answers the following questions:

* What is image classification?
* What is the purpose of the training, validation, and test sets and why do we need all of them?
* How do nearest neighbor and linear classifiers work?

Also include your results obtained from `simple_cats_dogs.py`. Include the validation accuracies for the different parameters you considered as a table or (better) a plot as well as the final test accuracy. Compare the best validation accuracy and the final test accuracy, and discuss the results.

## Submission

Submit your assignment until **April 15th at 11pm**. To do so, create a zip archive including the report, the complete `dlvc` folder with your implementations as well as `simple_cats_dogs.py`. More precisely, after extracting the archive we should obtain the following:

    group_x/
        report.pdf
        simple_cats_dogs.py
        dlvc/
            batches.py
            ...
            datasets/
                ...
            ...

Submit the zip archive in TUWEL. Make sure you've read the general assignment information [here](https://smithers.cvl.tuwien.ac.at/theitzinger/dlvc_ss21_public/-/blob/master/assignments/general.md) before your final submission.