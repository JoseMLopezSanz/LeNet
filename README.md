# LeNet
Implementation of the LeNet CNN for the MNIST dataset

This repository shows a simple implementation in keras of the CNN LeNet first introduced by 
LeCun et al. in their 1998 paper, Gradient-Based Learning Applied to Document Recognition.

lenet_mnist.py allows the user to train the network on the MNIST dataset and create a model
MNIST_model.hdf5 which can be used later in mnist_draw.py to test the performance of the
network manually. 

mnist_draw.py shows a blackboard where the user can write numbers with the mouse and obtain
a prediction for the handwritten number. 

In the future I'd like to train the network for the EMNIST dataset and be able to obtain 
handwritten recognition of alphanumeric characters in addition to only numbers. The same
blackboard implementation will be used to test the performance manually.

This code is really good for new students who want to get their hands dirty and learn how to
train the "Hello world" version of a CNN.

### Prerequisites

In order to use this code you'll need the following python libraries
Markup : * keras
         * cv2
*sklearn
*numpy
*matplotlib
*os

