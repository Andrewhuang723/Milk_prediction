import pandas as pd
import numpy as np
import torch
import datetime
import re

birth = pd.read_csv("./data/birth.csv")
label_data = birth[["9",  "10"]]
test = birth.loc[:, "3"].dropna()
print(birth.shape, test.shape)
label_data = label_data.loc[test.index, :].fillna(-1)

def unique(list1):
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

def one_hot_k(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def label_featurize(data, lab_feats):
    tot = []
    for row in data:
        row_feats = []
        for i, feats in zip(row, lab_feats):
            row_feats += one_hot_k(i, feats)
        if len(lab_feats) < len(row): #(狀況類別2~10 belong to the same label)
            for i in row[len(lab_feats):]:
                row_feats += one_hot_k(i, lab_feats[-1])
        tot.append(row_feats)
    return np.array(tot)

lab_feats = [unique(i) for i in label_data.values.T]
label_data = label_featurize(label_data.values, lab_feats)

birth_day = birth.loc[test.index, "2"].dropna()
birth_day = pd.to_datetime(birth_day)
int_birth_day = np.datetime64(birth_day)
label_data = np.concatenate([label_data, int_birth_day], axis=1)
test = pd.to_datetime(test)
interval = (birth_day - test) / np.timedelta64(1, "D")
interval = np.abs(interval.values)
print(interval.mean())

label_data = torch.Tensor(label_data).cuda()
interval = torch.Tensor(interval / np.max(interval)).cuda()

from milk.Regression_model import Model, Data, DataLoader
from torch import optim
import torch.nn as nn
from milk.pytorch_tools import EarlyStopping
from time import time

train_loader = DataLoader(dataset=Data(label_data, interval), batch_size=128, shuffle=True)
net = Model(n_features=label_data.shape[-1], n_hidden=200).cuda()
optimizer = optim.Adam(net.parameters(), lr=.0001)
criterion = nn.L1Loss().cuda()
early_stopping = EarlyStopping(patience=10)

def train(model, epochs, loss_func, data_loader, val_data_loader=None):
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
            y_pred = model.regression(X)
            loss = 0
            for i in range(len(y)):
                loss += loss_func(y_pred[i, :], y[i, :])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_epochs_loss += (loss / len(X))
        train_epochs_loss /= len(data_loader)
        if epoch >= 1:
            dur.append(time() - t0)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        print('Epoch {} | loss {:.6f} | Time(s) {:.4f}'.format(epoch, train_epochs_loss,
                                                                                 np.mean(dur)))
        train_epochs_losses.append(train_epochs_loss)

        dict = model.state_dict()
        dict["loss"] = train_epochs_losses
        dict["val_loss"] = val_epoch_losses
    return dict

model = train(model=net, epochs=20, loss_func=criterion, data_loader=train_loader)
torch.save(model, "dry_milk.pkl")
