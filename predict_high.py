import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import my_utils as ut

data = np.load('data.npy')
print(data)

