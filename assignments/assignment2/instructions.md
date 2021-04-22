
# Deep Learning for Visual Computing - Assignment 2

The second assignment covers iterative optimization and parametric (deep) models for image classification.

## Part 1

This part is about experimenting with different flavors of gradient descent and optimizers.

Download the data from [here](https://smithers.cvl.tuwien.ac.at/theitzinger/dlvc_ss21_public/-/tree/master/assignments/assignment2). Your task is to implement `optimizer_2d.py`. We will use various optimization methods implemented in PyTorch to find the minimum in a 2D function given as an image. In this scenario, the optimized weights are the coordinates at which the function is evaluated, and the loss is the function value at those coordinates.

See the code comments for instructions. The `fn/` folder contains sampled 2D functions for use with that script. For bonus points you can add and test your own functions (something interesting with a few local minima). For this you don't necessarily have to use `load_image`, you can also write a different function that generates a 2D array of values.

The goal of this part is for you to better understand the optimizers provided by PyTorch by playing around with them. Try different types (SGD, AdamW etc.), parameters, starting points, and functions. How many steps do different optimizers take to terminate? Is the global minimum reached? What happens when weight decay is set to a non-zero value and why? This nicely highlights the function and limitations of gradient descent, which we've already covered in the lecture.