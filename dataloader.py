from torch.utils.data import Dataset
import torch


# Write the override dataset class for our NN model
class MyDataset(Dataset):
    def __init__(self, data, args):
        self.feature = data[:, :-1, :-1]
        self.label = data[:, -1, -1]
        self.label = torch.where(self.label >= args.delta, 1.0, 0.0)

    def __getitem__(self, index):
        feature = self.feature[index]
        label = self.label[index]
        return {"feature": feature, "label": label}

    def __len__(self):
        return len(self.label)
