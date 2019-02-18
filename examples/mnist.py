import numpy as np
# import minpy.numpy as np

from tensorflow.examples.tutorials.mnist import input_data  # to download and read MNIST
import sys
import numpynet as nn
sys.path.append('../models')
from LeNet import LeNet
from ResNet18 import ResNet18

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

X_train = mnist.train.images.reshape([mnist.train.num_examples, 28, 28])
X_train = np.expand_dims(X_train, axis=1)
X_train = nn.Variable(X_train, lr=0)

Y_train = np.expand_dims(mnist.train.labels, axis=1)
Y_train = nn.onehot(Y_train)
Y_train = nn.Variable(Y_train, lr=0)

X_test = mnist.test.images.reshape([mnist.test.num_examples, 28, 28])
X_test = np.expand_dims(X_test, axis=1)
X_test = nn.Variable(X_test, lr=0)

Y_test = np.expand_dims(mnist.test.labels, axis=1)
Y_test = nn.onehot(Y_test)
Y_test = nn.Variable(Y_test, lr=0)

# lenet = LeNet(X_train.value.shape, 1024, 10)
# lenet.train(X_train, Y_train, X_test, Y_test, epochs=100, batchsize=100)

res18 = ResNet18(X_train.value.shape, 512, 10)
res18.train(X_train, Y_train, X_test, Y_test, epochs=10, batchsize=10)
