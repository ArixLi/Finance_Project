import numpy as np
import torch
import torch.nn as nn


# A simple linear regression model to predict the open and high price of a stock
# with historical data (at least 15 days, at most 40 days)
class SimpleLinearModel(nn.Module):
    def __init__(self,
                 input_size: int = 1,
                 hidden_size: int = 256,
                 output_size: int = 1):
        super(SimpleLinearModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lrelu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)
        self.lrelu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(hidden_size//2, hidden_size)
        self.lrelu3 = nn.LeakyReLU()
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.lrelu1(x)
        x = self.fc2(x)
        x = self.lrelu2(x)
        x = self.fc3(x)
        x = self.lrelu3(x)
        x = self.output_layer(x)
        return x


