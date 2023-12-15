import numpy as np
import torch
import torch.nn as nn


def split_data(data, args):
    """
    Split the data into fixed historical days
    :param data: Raw Price Data, shape: (len, 3) (High_Open_Ratio, Close_Open_Ratio, Open)
    :param args: the arguments
    :return: the training and validation sets
    """
    wrapped_data = []
    for i in range(0, len(data) - args.len_days - 1):
        cur_entity = data[i: i + args.len_days + 1]
        wrapped_data.append(cur_entity)
    wrapped_data = np.stack(wrapped_data)
    train_size = int(len(wrapped_data) * 0.7)
    train_data = wrapped_data[:train_size]
    val_data = wrapped_data[train_size:]

    return train_data, val_data


def profit_and_loss(pred, data, args):
    """
    Compute the profit and loss for a given High_Open_Ratio prediction. Note we multiply 100 for the ratio.
    :param pred: High_Open_Ratio prediction, shape: (len, )
    :param data: Raw Price Data, shape: (len, 3) (High_Open_Ratio, Close_Open_Ratio, Open)
    :param args: the arguments
    :return: the P&L results tot_PL
    """
    delta = args.delta
    threshold = args.threshold
    tot_PL = 0
    for i in range(pred.shape[0]):
        # When prediction is Long, compute the P&L with the true price
        if pred[i] >= threshold:
            if data[i, 0] >= delta:
                tot_PL += (data[i, 0] * data[i, 2] * 0.01)
            else:
                tot_PL += (data[i, 1] * data[i, 2] * 0.01)
    return tot_PL


# A simple linear regression model to predict the open and high price of a stock
# with historical data (at least 15 days, at most 40 days)
class SimpleLinearModel(nn.Module):
    def __init__(self,
                 input_size: int = 1,
                 days: int = 30,
                 hidden_size: int = 256,
                 output_size: int = 1):
        super(SimpleLinearModel, self).__init__()
        self.fc1 = nn.Linear(input_size * days, hidden_size)
        self.act1 = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.act2 = nn.Tanh()
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.act3 = nn.Tanh()
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    # Note that our output is the predicted P(x >= High_Open_Ratio)
    def forward(self, x):
        cur_bs = x.size(0)
        x = x.view(cur_bs, -1).contiguous()
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        return self.sigmoid(self.output_layer(x))


def pca(x, min_explained_var=0.99):
    """
    Perform Principal Component Analysis (PCA) on the input data.

    Parameters:
    - x: Input data matrix of shape (n_samples, n_features).
    - min_explained_var: Minimum cumulative explained variance ratio to retain.

    Returns:
    - transformed_data: Transformed data matrix.
    - components: Principal components (eigenvectors).
    - explained_var: Explained variance of each principal component.
    """
    # Center the data by subtracting the mean of each feature
    mean_x = np.mean(x, axis=0)
    centered_x = x - mean_x

    # Calculate the covariance matrix
    cov_matrix = np.cov(centered_x, rowvar=False)

    # Calculate eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvectors by decreasing eigenvalues
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Normalize eigenvalues to get explained variance ratios
    explained_var = eigenvalues[sorted_indices] / np.sum(eigenvalues)

    # Determine the number of components to retain
    cum_explained_var = np.cumsum(explained_var)
    num_components = np.argmax(cum_explained_var >= min_explained_var) + 1

    # Retain only the top 'num_components' components
    eigenvectors = eigenvectors[:, :num_components]

    # Project the data onto the new basis
    transformed_data = centered_x.dot(eigenvectors)

    return transformed_data, eigenvectors, explained_var[:num_components]