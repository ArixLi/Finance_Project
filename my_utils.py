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
    for i in range(args.len_days, len(data)-args.len_days-1):
        cur_entity = data[i:i + args.len_days+1]
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
    delta = args.threshold
    tot_PL = 0
    for i in range(pred.shape[0]):
        if pred[i] >= delta:
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
                 hidden_size: int = 256,
                 output_size: int = 1,
                 days: int = 30):
        super(SimpleLinearModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act1 = nn.Sigmoid()
        self.output_layer1 = nn.Linear(hidden_size, output_size)
        self.fc3 = nn.Linear(days, hidden_size)
        self.act3 = nn.Sigmoid()
        self.output_layer2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    # Note that our output is the predicted High_Open_Ratio
    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.output_layer1(x)
        x = x.squeeze(-1).contiguous()
        x = self.act3(self.fc3(x))
        return self.sigmoid(self.output_layer2(x))
