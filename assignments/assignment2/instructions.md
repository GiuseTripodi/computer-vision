
# Deep Learning for Visual Computing - Assignment 2

The second assignment covers iterative optimization and parametric (deep) models for image classification.

## Part 1

This part is about experimenting with different flavors of gradient descent and optimizers.

Download the data from [here](https://smithers.cvl.tuwien.ac.at/theitzinger/dlvc_ss21_public/-/tree/master/assignments/assignment2). Your task is to implement `optimizer_2d.py`. We will use various optimization methods implemented in PyTorch to find the minimum in a 2D function given as an image. In this scenario, the optimized weights are the coordinates at which the function is evaluated, and the loss is the function value at those coordinates.

See the code comments for instructions. The `fn/` folder contains sampled 2D functions for use with that script. For bonus points you can add and test your own functions (something interesting with a few local minima). For this you don't necessarily have to use `load_image`, you can also write a different function that generates a 2D array of values.

The goal of this part is for you to better understand the optimizers provided by PyTorch by playing around with them. Try different types (SGD, AdamW etc.), parameters, starting points, and functions. How many steps do different optimizers take to terminate? Is the global minimum reached? What happens when weight decay is set to a non-zero value and why? This nicely highlights the function and limitations of gradient descent, which we've already covered in the lecture.

## Part 2

Time for some Deep Learning. We already implemented most of the required functionality during Assignment 1. Make sure to fix any mistakes mentioned in the feedback you received for your submission. With the exception of `simple.py` and `simple_cats_dogs.py` all files will be reused in this assignment. The main thing that's missing is a subtype of `Model` that wraps a PyTorch CNN classifier. Implement this type, which is defined inside `dlvc/models/pytorch.py` and named `CnnClassifier`. Details are stated in the code comments. The PyTorch documentation of `nn.Module`, which is the base class of PyTorch models, is available [here](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module).

PyTorch (and other libraries) expects the channel dimension of a single sample to be the first one, rows the second one, and columns the third one (`CHW` for short). However in our case they are `HWC`. To address this, implement the `hwc2chw()` function in `ops.py` (make sure to download the updated reference code).

Once this is in place, create a script named `cnn_cats_dogs.py`. This file will be very similar to the version for the simple classifier developed for Assignment 1 so you might want to use that one as a reference. This file should implement the following in the given order:

1. Load the training and validation subsets of the `PetsDataset`.
2. Initialize `BatchGenerator`s for both with batch sizes of 128 or so (feel free to experiment) and the input transformations required for the CNN. This should include input normalization. A basic option is `ops.add(-127.5), ops.mul(1/127.5)` but for bonus points you can also experiment with more sophisticated alternatives such as per-channel normalization using statistics from the training set (if so create corresponding operations in `ops.py` and document your findings in the report).
3. Define a PyTorch CNN with an architecture suitable for cat/dog classification. To do so create a subtype of `nn.Module` and overwrite the `__init__()` and `forward()` methods (do this inside `cnn_cats_dogs.py`). If you have access to an Nvidia GPU transfer the model using the `.cuda()` method of the CNN object.
4. Wrap the CNN object `net` in a `CnnClassifier`, `clf = CnnClassifier(net, ...)`.
5. Inside a `for epoch in range(100):` loop (i.e. train for 100 epochs which is sufficient for now), train `clf` on the training set and store the losses returned by `clf.train()` in a list. Then convert this list to a numpy array and print the mean and standard deviation in the format `mean ± std`. Then print the accuracy on the validation set using the `Accuracy` class developed in Assignment 1.

The console output should thus be similar to the following (ignoring the values):

    epoch 1
     train loss: 0.689 ± 0.006
     val acc: accuracy: 0.561
    epoch 2
     train loss: 0.681 ± 0.008
     val acc: accuracy: 0.578
    epoch 3
     train loss: 0.673 ± 0.009
     val acc: accuracy: 0.585
    epoch 4
     train loss: 0.665 ± 0.013
     val acc: accuracy: 0.594
    epoch 5
     train loss: 0.658 ± 0.014
     val acc: accuracy: 0.606
    ...

The goal of this part is for you to get familiar with PyTorch and to be able to try out different architectures and layer combinations. The pets dataset is ideal for this purpose because it is small. Experiment with the model by editing the code manually rather than automatically via hyperparameter optimization. What you will find is that the training loss will approach 0 even with simple architectures (demonstrating how powerful CNNs are and how well SGD works with them) while the validation accuracy will likely not exceed 75%. The latter is due to the small dataset size, resulting in overfitting. We will address this in the next part.

## Server Usage

You may find that training is slow on your computer unless you have an Nvidia GPU with CUDA support. If so, copy the code into your home directory on the DLVC server and run it there. You should have already received login credentials on April 9th - check your spam folder if you didn't. For details on how to run your scripts see [here](https://smithers.cvl.tuwien.ac.at/theitzinger/dlvc_ss21_public/-/tree/master/assignments/DLVC2021Guide.pdf). For technical problems regarding our server please contact [email](mailto:dlvc-trouble@cvl.tuwien.ac.at).

We expect queues will fill up close to assignment deadlines. In this case, you might have to wait a long time before your script even starts. In order to minimize wait times, please do the following:

* Write and test your code locally on your system. If you have a decent Nvidia GPU, please train locally and don't use the servers. If you don't have such a GPU, perform training for a few epochs on the CPU to ensure that your code works. If this is the case, upload your code to our server and do a full training run there. To facilitate this process, have a variable or a runtime argument in your script that controls whether CUDA should be used. Disable this locally and enable it on the server.
* Don't schedule multiple training runs in a single job, and don't submit multiple long jobs. Be fair.
* If you want to train on the server, do so as early as possible. If everyone starts two days before the deadline, there will be long queues and your job might not finish soon enough.

## Submission

The *tentative* deadline for this assignment is **May 27th at 11pm**. Depending on how fast the lecture is progressing, it may be pushed back. The last part of this assignment will be part 3.