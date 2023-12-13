import torch
import numpy as np
import my_utils as ut


# Get the PCA for the data set
raw_data = np.load('dataset/data_vix.npy')
invariants = raw_data[:, -3:]
raw_data = raw_data[:, :-3]

transformed_data, components, explained_var = ut.pca(raw_data)
data = np.hstack([transformed_data, invariants])

np.save('dataset/pca_data_vix.npy', data)
