import h5py
import numpy as np

num_neurons = 128
num_hidden_layers = 1
final_sparsity = 0.9

keras_file = 'model/KERAS_mnist_mlp_weights.h5'
pruned_keras_file = 'model/KERAS_mnist_mlp%i_prune%.2f_weights.h5'%(num_neurons,final_sparsity)
f = h5py.File(pruned_keras_file,'r')

total_weights = 0
total_pruned = 0

print('pruning summary')
for i in range(1,num_hidden_layers+1):
    weight = f['dense_{i}/dense_{i}/kernel:0'.format(i=i)][()]
    weight = weight.reshape(-1)
    weights = len(weight)
    pruned = weights - len(weight[np.abs(weight)>0])
    total_weights += weights
    total_pruned += pruned
    perc_pruned = pruned*100./weights
    print('  layer %i: %.1f%% weights pruned'%(i,perc_pruned))

print('model: %.1f%% weights pruned'%(total_pruned*100./total_weights))
