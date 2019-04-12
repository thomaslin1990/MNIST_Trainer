import math
import numpy as np
import chainer
from chainer import backend
from chainer import backends
from chainer.backends import cuda
from chainer import Function, FunctionNode, gradient_check, report, training, utils, Variable
from chainer import datasets, initializers, iterators, optimizers, serializers
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer import iterators, Chain
from chainer.datasets import mnist
import ipdb

train, test = mnist.get_mnist(withlabel=True, ndim=1)
# Choose the minibatch size
batchsize = 128

train_iter = iterators.SerialIterator(train, batchsize)
test_iter = iterators.SerialIterator(test, batchsize, repeat=False, shuffle=False)

class MyMLP(Chain):
    def __init__(self, n_hidden = 100, n_output = 10):
        super(MyMLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_hidden)
            self.l2 = L.Linear(n_hidden, n_hidden)
            self.l3 = L.Linear(n_hidden, n_output)


    def forward(self, input):
        h = self.l1(input)
        h = F.relu(h)
        h = self.l2(h)
        h = F.relu(h)
        return self.l3(h)



model = MyMLP()
gpu_id = 0
if gpu_id >= 0:
    model.to_gpu(gpu_id)

from chainer import optimizers

optimizer = optimizers.MomentumSGD(lr = 0.01, momentum=0.9)
optimizer.setup(model)


import numpy as np
from chainer.dataset import concat_examples
from chainer.backends.cuda import to_cpu

if __name__ == '__main__':

    max_epochs = 10

    while train_iter.epoch < max_epochs:

        #----------------one iteration training start from here-----
        train_batch = train_iter.next()
        image_train, target_train = concat_examples(train_batch, gpu_id)

        # calculate the prediction of the network
        prediction_train = model(image_train)

        # calculate the loss with softmax_cross_entropy
        loss = F.softmax_cross_entropy(prediction_train, target_train)

        # calculate the gradients in the network
        model.cleargrads()
        loss.backward()

        # update all the trainable parameters
        optimizer.update()

        #----------------finish one iteration tuning---------

        # check the validation accuracy
        if train_iter.is_new_epoch: # if this is the final minibatch size of the epoch
            # Display the training loss
            print('epoch:{:2d} train_loss:{:.04f}'.format(train_iter.epoch, float(to_cpu(loss.array))), end='')

            test_losses = []
            test_accuracies = []
            while True:
                test_batch = test_iter.next()
                image_test, target_test = concat_examples(test_batch, gpu_id)

                # Forward the test data
                prediction_test = model(image_test)

                # calculate the loss:
                loss_test = F.softmax_cross_entropy(prediction_test, target_test)
                test_losses.append(to_cpu(loss_test.array))

                # calcualte the accuracy
                accuracy = F.accuracy(prediction_test, target_test)
                accuracy.to_cpu()
                test_accuracies.append(accuracy.array)

                if test_iter.is_new_epoch:
                    test_iter.epoch = 0
                    test_iter.current_position = 0
                    test_iter.is_new_epoch = False
                    test_iter._pushed_position = None
                    break

            print('val_loss:{:.04f} val_accuracy:{:.4f}'.format(np.mean(test_losses), np.mean(test_accuracies)))
            serializers.save_npz('my_mnist.model', model)




