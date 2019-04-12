from chainer import serializers
from MNIST_Using_Training_Loop import MyMLP,test
import matplotlib.pyplot as plt

model = MyMLP()

serializers.load_npz('my_mnist.model', model)

x,t = test[300]
plt.imshow(x.reshape(28, 28), cmap='gray')
#plt.savefig('test.png')

print(x.shape, end='->')
x = x[None,...]
print(x.shape)

y = model(x)
y = y.array

pred_label = y.argmax(axis=1)[0]
print("predicted label:", pred_label)
plt.show()

