import h5py
import numpy as np

weight_names = []
def get_weight_names(name, obj):
    if 'kernel' in name: 
        weight_names.append(name)

num_neurons = 128
num_hidden_layers = 1
final_sparsity = 0.9

keras_file = 'model/KERAS_mnist_mlp_weights.h5'
pruned_keras_file = 'model/KERAS_mnist_mlp%i_prune%.2f_weights.h5'%(num_neurons,final_sparsity)
f = h5py.File(pruned_keras_file,'r')
f.visititems(get_weight_names)

total_weights = 0
total_pruned = 0

print('pruning summary')
for weight_name in weight_names:
    weight = f[weight_name][()]
    # reshape to be flat
    weight = weight.reshape(-1)
    weights = len(weight)
    pruned = weights - len(weight[np.abs(weight)>0])
    total_weights += weights
    total_pruned += pruned
    perc_pruned = pruned*100./weights
    print('  layer %s: %.1f%% weights pruned'%(weight_name,perc_pruned))

print('model: %.1f%% weights pruned'%(total_pruned*100./total_weights))
