import math
import numpy as np
import chainer
from chainer import backend
from chainer import backends
from chainer.backends import cuda
from chainer import Function, FunctionNode, gradient_check, report, training, utils, Variable
from chainer import datasets, initializers, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.links as L
import chainer.functions as F
from chainer.training import extensions

# Lenet Structure

class LeNet5(Chain):
    def __init__(self):
        super(LeNet5, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(in_channels=1, out_channels=6, ksize=5, stride=1)
            self.conv2 = L.Convolution2D(in_channels=6, out_channels=16, ksize=5, stride=1)
            self.conv3 = L.Convolution2D(in_channels=16, out_channels=120, ksize=4, stride=1)
            self.fc4 = L.Linear(None, 84)
            self.fc5 = L.Linear(84, 10)

    def forward(self, x):
        h = F.sigmoid(self.conv1(x))
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = F.sigmoid(self.conv2(h))
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = F.sigmoid(self.conv3(h))
        h = F.sigmoid(self.fc4(h))
        h = self.fc5(h)

        if chainer.config.train:
            return h
        else:
            return F.softmax(h)


if __name__ == '__main__':
    model = LeNet5()

    # Input and label
    x = np.random.randn(32, 1, 28, 28).astype(np.float32)
    t = np.random.randint(0, 10, size=(32,)).astype(np.int32)

    y = model(x)
    loss = F.softmax_cross_entropy(y, t)
    print("loss:", loss.array)




