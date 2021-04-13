import pandas as pd
import torch
import numpy as np
from sklearn.decomposition import PCA
import pickle


birth = pd.read_csv("data/birth.csv", index_col=None)
breed = pd.read_csv("data/breed.csv", index_col=None)
data1 = pd.read_csv("data/report.csv").fillna(-1)
spec = pd.read_csv("data/spec.csv", index_col=None)
submission = pd.read_csv("data/submission.csv")


def unique(list1):
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

#1. Labels
report = data1.values
label= [1, 2, 3, 8, 17]
lab_feats = [unique(report[:,1]), unique(report[:,2]), unique(report[:,3]), unique(report[:,8]), unique(report[:,17])]

def one_hot_k(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def label_featurize(row):
    s = []
    for i, feats in zip(label, lab_feats):
        s += one_hot_k(row[i], feats)
    return s

for i in range(report.shape[0]):
    if i == 0:
        row = np.array(label_featurize(report[i])).reshape(1,-1)
    else:
        row = np.concatenate([row, np.array(label_featurize(report[i])).reshape(1, -1)], axis=0)
print(row.shape)

#2. Sperm labels
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
# TSNE convert sperm features into 2 features (saved)
min_max_scaler = MinMaxScaler()
std_scaler = StandardScaler()

label_df = pd.DataFrame(row)
print(label_df.shape)

#3. Time interval
cow_data = pd.read_csv("./prediction.csv")

con_df = data1[["10", "14"]].values
con_df = pd.DataFrame(con_df, columns=["泌乳天數", "月齡"])

re_data = pd.concat([label_df, con_df], axis=1)

check_data = cow_data[["檢測日期"]].values
norm_3 = min_max_scaler.fit_transform(np.abs(check_data))

pick_data = cow_data[["採樣日期"]].values
norm_4 = min_max_scaler.fit_transform(np.abs(pick_data))

latest_birth = cow_data[["最近分娩日期"]].values
norm_5 = min_max_scaler.fit_transform(np.abs(latest_birth))

latest_mate = cow_data[["最後配種日期"]].values

re_data = pd.concat([re_data, pd.DataFrame(norm_3, columns=["檢測日期"]),
                     pd.DataFrame(norm_4, columns=["採樣日期"]), pd.DataFrame(norm_5, columns=["最近分娩日期"]), cow_data[["乳量"]].fillna(-1)], axis=1)

re_data = re_data.dropna()
re_data.to_csv("./re_data_2.csv")
print(re_data.shape)
'''
feat = re_data.values[:, :-6]
reg = re_data.values[:, -6:]
n_components = 20
pca_d = PCA(n_components=n_components)
pca_d.fit(feat)
print(np.sum(pca_d.explained_variance_ratio_))
n_pca = pca_d.fit_transform(feat)

re_data = np.concatenate([n_pca, reg], axis=1)
re_data = pd.DataFrame(re_data, columns=list(range(n_components)) + list(range(reg.shape[-1] - 1)) + ["乳量"])
'''
#re_data = pd.concat([re_data, cow_data[["乳量"]].fillna(-1)], axis=1)
train = re_data[re_data["乳量"] >= 0]
test = re_data[re_data["乳量"] < 0]
print("train: ", train.shape, "test: ", test.shape)
print(re_data)
X_train = train.drop(columns=["乳量"], )
X_train = torch.Tensor(list(X_train.values)).cuda()

y_train = train[["乳量"]]
y_train = torch.Tensor(list(y_train.values)).cuda()

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=0)

X_test = test.drop(columns=["乳量"])
X_test = torch.Tensor(list(X_test.values)).cuda()

torch.save(X_test, "./data/X_test.pkl")
print("X_train: ", X_train.shape, "y_train: ", y_train.shape,
      "X_val: ", X_val.shape, "X_test: ", X_test.shape)

from torch import nn
from torch import optim
from torch.nn import MSELoss
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from time import time
from milk.Regression_model import Model, Data
from milk.pytorch_tools import EarlyStopping

train_loader = DataLoader(dataset=Data(X_train, y_train), batch_size=128, shuffle=False)
val_loader = DataLoader(dataset=Data(X_val, y_val), batch_size=128, shuffle=False)

net = Model(n_features=X_train.shape[-1], n_hidden=200).cuda()
print(summary(net, input_size=(X_train.shape[-1],)))
optimizer = optim.Adam(net.parameters(), lr=.01)
criterion = nn.L1Loss().cuda()
early_stopping = EarlyStopping(patience=20)

def train(model, epochs, loss_func, data_loader, val_data_loader):
    train_epochs_losses = []
    val_epoch_losses = []
    dur = []
    for epoch in range(epochs):
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

model = train(model=net, epochs=100, loss_func=criterion, data_loader=train_loader, val_data_loader=val_loader)

torch.save(model, "./Model.pkl")