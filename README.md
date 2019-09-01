# MNIST MLP

## Source
https://github.com/keras-team/keras/blob/master/examples/mnist_mlp.py

## Quick Start
To setup with conda/pip (assuming they already installed):
```
source setup.sh
```

To train the model and generate `.h5` and `.json` files:
```
make train
```

To test the model:
```
make prediction
```

To prune the model to 90% sparsity:
```
python mnist_mlp_prune.py
```

To check the model compression:
```
python check_compression.py
```

Compare the local implementation with the source:
```
make diff
```

To run a hyperparameter scan with guild.ai:
```
guild run train epochs=100 optimizer=[adam,nadam,rmsprop,sgd] dropout_rate=[0,0.1,0.2] l1_reg=[0,1e-5,1e-4]
```

To run different pruning hyperparameters:
```
guild run prune epochs=100 optimizer=[adam,nadam,rmsprop,sgd] final_sparsity=[0.3,0.5,0.7,0.9,0.95,0.97,0.99]
```