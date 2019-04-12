import math
import numpy as np
import chainer
from chainer import backend
from chainer import backends
from chainer.backends import cuda
from chainer import Function, FunctionNode, gradient_check, report, training, utils, Variable
from chainer import datasets, initializers, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

from chainer.datasets import mnist
train, test = mnist.get_mnist()

batchsize = 128
train_iter = iterators.SerialIterator(train, batchsize)
test_iter = iterators.SerialIterator(test, batchsize, False, False)

# model definition
class MLP(Chain):
    def __init__(self, n_mid_units = 100, n_out = 10):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_mid_units)
            self.l2 = L.Linear(n_mid_units, n_mid_units)
            self.l3 = L.Linear(n_mid_units, n_out)

    def forward(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

if __name__ == '__main__':

    gpu_id = 0

    model = MLP()
    if gpu_id >= 0:
        model.to_gpu(gpu_id)

    # Create Updater
    max_epoch = 10

    # default is 'softmax cross entropy loss'
    model = L.Classifier(model)

    # optimier setup
    optimizer = optimizers.MomentumSGD()

    # Give the optimizer a reference to the model
    optimizer.setup(model)

    # Get an updater that uses the Iterator and Optimizer
    updater = training.updaters.StandardUpdater(train_iter, optimizer, device=gpu_id)

    # Setup a trainer
    trainer = training.Trainer(updater, (max_epoch, "epoch"), out='mnist_result')

    # Add Extensions to the Trainer object
    from chainer.training import extensions

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
    trainer.extend(extensions.snapshot_object(model.predictor, filename='model_epoch-{.updater.epoch}'))
    trainer.extend(extensions.Evaluator(test_iter, model, device=gpu_id))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss'
                                           , 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name = 'accuracy.png'))
    trainer.extend(extensions.dump_graph('main/loss'))


    trainer.run()




