'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function
import tensorflow as tf
import tensorboard
import tensorflow.keras as keras

from tensorflow.keras.models import model_from_json
from tensorflow.keras.datasets import mnist
from tensorflow_model_optimization.sparsity import keras as sparsity

import numpy as np

import tempfile
logdir = tempfile.mkdtemp()

batch_size = 128
num_neurons = 128
num_classes = 10
num_inputs = 28*28
epochs = 20
initial_sparsity = 0.0
final_sparsity = 0.9

# the data, split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, num_inputs)
X_test = X_test.reshape(-1, num_inputs)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
num_train_samples = X_train.shape[0]
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Model reconstruction from JSON file
with open('./model/KERAS_mnist_mlp%s.json'%num_neurons, 'r') as f:
    model = model_from_json(f.read())

# Load weights into the new model
model.load_weights('./model/KERAS_mnist_mlp%s_weights.h5'%num_neurons)

model.summary()

end_step = np.ceil(1.0 * num_train_samples / batch_size).astype(np.int32) * epochs
print(end_step)

new_pruning_params = {
    'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=initial_sparsity,
                                                 final_sparsity=final_sparsity,
                                                 begin_step=0,
                                                 end_step=end_step,
                                                 frequency=100)
}

pruned_model = sparsity.prune_low_magnitude(model, **new_pruning_params)
pruned_model.summary()

pruned_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

callbacks = [tf.keras.callbacks.TensorBoard(log_dir=logdir, profile_batch=0),
             sparsity.UpdatePruningStep(),
             sparsity.PruningSummaries(log_dir=logdir, profile_batch=0)
]
history = pruned_model.fit(X_train, y_train,
                           batch_size=batch_size,
                           epochs=epochs,
                           verbose=1,
                           validation_data=(X_test, y_test),
                           callbacks=callbacks)
score = pruned_model.evaluate(X_test, y_test, verbose=0)


print('Test loss:', score[0])
print('Test accuracy:', score[1])

pruned_model = sparsity.strip_pruning(pruned_model)
pruned_model.summary()
#print_model_to_json(pruned_model, 'model/KERAS_mnist_mlp.json')
pruned_model.save_weights('model/KERAS_mnist_mlp%i_weights_prune%.2f.h5'%(num_neurons,final_sparsity))

