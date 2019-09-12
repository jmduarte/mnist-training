'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function
import tensorflow as tf
import tensorboard
import tensorflow.keras as keras

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import regularizers

import tempfile
logdir = tempfile.mkdtemp()

import os
os.makedirs('model',exist_ok=True)

batch_size = 128
num_neurons = 128
num_classes = 10
num_inputs = 28*28
epochs = 20
dropout_rate = 0.1
optimizer = 'adam'
l1_reg = 0 # 0.0001

# the data, split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, num_inputs)
X_test = X_test.reshape(-1, num_inputs)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

inputs = Input(shape = (num_inputs,), name='input_1')
x = Dense(num_neurons, activation='relu', name='dense_1', kernel_regularizer=regularizers.l1(l1_reg))(inputs)
x = Dropout(dropout_rate, name='dropout_1')(x)
x = Dense(num_neurons, activation='relu', name='dense_2', kernel_regularizer=regularizers.l1(l1_reg))(x)
x = Dropout(dropout_rate, name='dropout_2')(x)
x = Dense(num_neurons, activation='relu', name='dense_3', kernel_regularizer=regularizers.l1(l1_reg))(x)
x = Dropout(dropout_rate, name='dropout_3')(x)
x = Dense(num_neurons, activation='relu', name='dense_4', kernel_regularizer=regularizers.l1(l1_reg))(x)
x = Dropout(dropout_rate, name='dropout_4')(x)
x = Dense(num_neurons, activation='relu', name='dense_5', kernel_regularizer=regularizers.l1(l1_reg))(x)
x = Dropout(dropout_rate, name='dropout_5')(x)
outputs = Dense(num_classes, activation='softmax', name='dense_6', kernel_regularizer=regularizers.l1(l1_reg))(x)

model = Model(inputs=inputs, outputs=outputs)

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

callbacks = [tf.keras.callbacks.TensorBoard(log_dir=logdir, profile_batch=0),
             tf.keras.callbacks.ModelCheckpoint('model/KERAS_mnist_mlp%i_weights.h5'%num_neurons, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1),
             tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=20)
             ]

history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_test, y_test),
                    callbacks=callbacks)

# reload best weights
model.load_weights('model/KERAS_mnist_mlp%i_weights.h5'%num_neurons)
score = model.evaluate(X_test, y_test, verbose=0)

best_test_loss = score[0]
best_test_acc = score[1]

print('best_test_loss: %f'% best_test_loss)
print('best_test_acc: %f'% best_test_acc)

def print_model_to_json(keras_model, outfile_name):
    outfile = open(outfile_name,'w')
    jsonString = keras_model.to_json()
    import json
    with outfile:
        obj = json.loads(jsonString)
        json.dump(obj, outfile, sort_keys=True,indent=4, separators=(',', ': '))
        outfile.write('\n')

print_model_to_json(model, 'model/KERAS_mnist_mlp%i.json'%num_neurons)
#model.save_weights('model/KERAS_mnist_mlp%i_weights.h5'%num_neurons)

