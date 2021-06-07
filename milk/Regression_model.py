import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from time import time
from milk.pytorch_tools import EarlyStopping

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


early_stopping = EarlyStopping(patience=10)

def train(model, epochs, loss_func, data_loader, optimizer,  val_data_loader=None):
    train_epochs_losses = []
    val_epoch_losses = []
    dur = []


    for epoch in range(epochs):
        model.train()
        train_epochs_loss = 0
        if epoch >= 1:
            t0 = time()
        for X, y in data_loader:
            X = X.cuda()
            y = y.cuda()
            y = y.view(-1, 1)
            y_pred = model(X)
            loss = 0
            for i in range(len(y)):
                loss += loss_func(y_pred[i, :], y[i, :])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_epochs_loss += (loss / len(X))
        train_epochs_loss /= len(data_loader)

        val_epoch_loss = 0
        model.eval()
        for X_val, y_val in val_data_loader:
            X_val = X_val.cuda()
            y_val = y_val.cuda()
            y_val = y_val.view(-1, 1)
            y_val_pred = model(X_val)
            val_loss = 0
            for i in range(len(y_val)):
                val_loss += loss_func(y_val_pred[i, :], y_val[i, :])
            val_epoch_loss += (val_loss / len(y_val))
        val_epoch_loss /= len(val_data_loader)

        if epoch >= 1:
            dur.append(time() - t0)

        early_stopping(val_loss=val_epoch_loss, model=model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        print('Epoch {} | loss {:.6f} | Time(s) {:.4f} | val_loss {:.6f}'.format(epoch, train_epochs_loss,
                                                                                 np.mean(dur),
                                                                                 val_epoch_loss))
        train_epochs_losses.append(train_epochs_loss)
        val_epoch_losses.append(val_epoch_loss)

        dict = model.state_dict()
        dict["loss"] = train_epochs_losses
        dict["val_loss"] = val_epoch_losses
    return dict
