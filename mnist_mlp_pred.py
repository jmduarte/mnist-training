from __future__ import print_function
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.models import model_from_json
from tensorflow.keras.datasets import mnist

import matplotlib.pyplot as plt
import numpy as np

x_input = 28
y_input = 28
num_neurons = 128
num_inputs = x_input*y_input
num_classes = 10
model_name = './model/KERAS_mnist_mlp%i'%num_neurons
#final_sparsity = 0.9
#model_name = './model/KERAS_mnist_mlp%i_prune%.2f'%(num_neurons,final_sparsity)

import sys

# Model reconstruction from JSON file
with open('%s.json'%model_name, 'r') as f:
    model = model_from_json(f.read())

# Load weights into the new model
model.load_weights('%s_weights.h5'%model_name)

model.compile(loss='categorical_crossentropy',
                            optimizer='adam',
                            metrics=['accuracy'])


# The data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_test = x_test.reshape(-1, num_inputs)
x_test = x_test.astype('float32')
x_test /= 255

# Convert class vectors to binary class matrices
y_test = keras.utils.to_categorical(y_test, num_classes)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Choose an input image
image_index = 133 # up to 60,000
image = x_test[image_index]
label = y_test[image_index]

# Run prediction
pred = model.predict(np.array([image.reshape(num_inputs,)]))

# Some information
print('INFO: input shape: ', image.shape)
print('INFO: image shape: ', image.reshape(x_input, y_input).shape)
print('INFO: one-hot encoding: ', label) # 2
print('INFO: predictions: ', pred[0])
print('INFO: top prediction: ', pred.argmax())

# Show image
plt.imshow(x_test[image_index].reshape(x_input, y_input), cmap='Greys')
plt.show()

np.savetxt('%s_input_features.dat'%model_name, x_test, fmt='%g',)
np.savetxt('%s_truth_labels.dat'%model_name, y_test, fmt='%g',)
np.savetxt('%s_predictions.dat'%model_name, pred, fmt='%g',)
