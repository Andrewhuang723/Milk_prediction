import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader


class Model(nn.Module):
    def __init__(self, n_features, n_hidden):
        super(Model, self).__init__()
        self.n_features = n_features
        self.lin_0 = nn.Linear(n_features, n_hidden)
        self.lin_1 = nn.Linear(n_hidden, n_hidden)
        self.predict = nn.Linear(n_hidden, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs):
        hidden = nn.ReLU()(self.lin_0(inputs))
        hidden = self.dropout(hidden)
        hidden = nn.ReLU()(self.lin_1(hidden))
        hidden = nn.ReLU()(self.lin_1(hidden))
        out = nn.ReLU()(self.predict(hidden))
        return out


class Data(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return self.X[item], self.y[item]