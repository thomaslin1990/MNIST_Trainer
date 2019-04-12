import matplotlib.pyplot as plt
from MNIST_Using_Trainer import *
import ipdb

model = MLP()
serializers.load_npz('mnist_result/model_epoch-10', model)

# Show the output
x, t = test[100]
plt.imshow(x.reshape(28, 28), cmap='gray')
print('label:', t)

print(x.shape)
print(x[np.newaxis,:].shape)

y = model(x[np.newaxis,:])
print('predicted_label', y.data.argmax(axis=1)[0])

plt.show()

