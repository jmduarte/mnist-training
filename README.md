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
python mnist_mlp.prune.py
```

To check the model compression:
```
python check_compression.py
```

Compare the local implementation with the source:
```
make diff
```
