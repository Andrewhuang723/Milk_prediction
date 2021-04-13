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
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
# TSNE convert sperm features into 2 features (saved)
'''
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
t_sne = TSNE(n_components=2, perplexity=30)
sperm_1 = row[:, 50:299]
sperm_2 = row[:, 299:]
n_sperm_1 = t_sne.fit_transform(sperm_1)
n_sperm_2 = t_sne.fit_transform(sperm_2)
n_sperm = np.concatenate([n_sperm_1, n_sperm_2], axis=1)
data = pd.DataFrame(n_sperm, columns=["sperm_feature_11", "sperm_feature_12", "sperm_feature_21", "sperm_feature_22"])
'''

min_max_scaler = MinMaxScaler()
std_scaler = StandardScaler()
'''
sperm_features = pd.read_excel("sperm_features.xlsx", index_col=None).values[:, 1:]
norm_sperm_features = std_scaler.fit_transform(sperm_features)
label = np.concatenate([row, norm_sperm_features], axis=1)
label_df = pd.DataFrame(label)
print(label_df.shape)
'''
#3. Time interval
latest_birth = data1.loc[:, "12"].values
past_birth = data1.loc[:, "19"].values
for i in range(len(past_birth)):
    if past_birth[i] == -1:
        past_birth[i] = latest_birth[i]
latest_birth = pd.Series(pd.to_datetime(latest_birth))
past_birth = pd.Series(pd.to_datetime(past_birth))
int_1 = (latest_birth - past_birth) / np.timedelta64(1, "D")
norm_1 = std_scaler.fit_transform(np.abs(np.array(int_1).reshape(-1, 1)))
birth_interval = pd.DataFrame(norm_1, columns=["分娩間隔"])

latest_mate = data1.loc[:, "16"]
first_mate = data1.loc[:, "20"]
latest_mate = pd.to_datetime(latest_mate)
first_mate = pd.to_datetime(first_mate)
int_2 = (latest_mate - first_mate) / np.timedelta64(1, "D")
norm_2 = std_scaler.fit_transform(np.abs(np.array(int_2).reshape(-1, 1)))
mate_interval = pd.DataFrame(norm_2, columns=["配種間隔"])

time_interval = pd.concat([birth_interval, mate_interval], axis=1) #(12, 16, 19, 20)

report_1 = time_interval

a_con_df = data1[["10", "14"]].values
norm_con_df = std_scaler.fit_transform(a_con_df)
con_df = pd.DataFrame(norm_con_df, columns=["泌乳天數", "月齡"])

re_data = pd.concat([report_1, con_df], axis=1)

#4. Cow_data
cow_data = pd.read_csv("./data/cow_data.csv")
data_feats = cow_data[["乳牛編號", "第一次配種精液", "最後配種精液", "分娩難易度", "配種方式", "精液種類", "狀況類別1"]].fillna(-1)
mis_ind = data_feats[data_feats["配種方式"] == -1].index
mis_ind_1 = data_feats[data_feats["精液種類"] == -1].index
mis_ind_2 = data_feats[data_feats["狀況類別1"] == "c"].index
data_feats.loc[mis_ind_1, "精液種類"] = "-1"
data_feats.loc[mis_ind, "配種方式"] = "-1"
data_feats.loc[mis_ind_2, "狀況類別1"] = "C"
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

lab_feats = [unique(i) for i in data_feats.values.T]
#data_feats = pd.concat([data_feats, cow_data[["狀況類別2", "狀況類別3", "狀況類別4", "狀況類別5", "狀況類別6", "狀況類別7", "狀況類別8", "狀況類別9", "狀況類別10"]].fillna(-1)], axis=1)

label_data = label_featurize(data_feats.values, lab_feats)
dry_milk_data = cow_data[["乾乳日期"]].fillna(0).values
norm_3 = std_scaler.fit_transform(np.abs(dry_milk_data))
re_data = pd.concat([pd.DataFrame(label_data), re_data, pd.DataFrame(norm_3, columns=["乾乳日期"])], axis=1)
print(re_data.shape)

#from sklearn.manifold import TSNE
#t_sne = TSNE(n_components=2, perplexity=30)
#n_tsne = t_sne.fit_transform(re_data)
#tsne_df = pd.DataFrame(n_tsne, columns=["n_1", "n_2"])
#tsne_df.to_csv("./tsne_label.csv")
n_tsne = pd.read_csv("./tsne_label.csv")[["n_1", "n_2"]]

re_data = pd.concat([n_tsne, data1[["11"]]], axis=1)
train = re_data[re_data["11"] >= 0]
test = re_data[re_data["11"] < 0]
print("train: ", train.shape, "test: ", test.shape)

X_train = train.drop(columns=["11"])
X_train = torch.Tensor(list(X_train.values)).cuda()

X_train = X_train.reshape()

y_train = train.loc[:, "11"]
y_train = torch.Tensor(list(y_train.values)).cuda()

def split(X, Y, rate):
    X_val = X[int(X.shape[0]*rate):]
    y_val = Y[int(Y.shape[0]*rate):]
    X_train = X[:int(X.shape[0]*rate)]
    y_train = Y[:int(Y.shape[0]*rate)]
    return X_train, y_train, X_val, y_val

X_train, y_train, X_val, y_val = split(X_train, y_train, rate=0.75)

X_test = test.drop(columns=["11"])
X_test = torch.Tensor(list(X_test.values)).cuda()

torch.save(X_test, "./data/X_test_tsne.pkl")
# no y_test
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

net = Model(n_features=X_train.shape[-1], n_hidden=100).cuda()
print(summary(net, input_size=(X_train.shape[-1],)))
optimizer = optim.Adam(net.parameters(), lr=.001, weight_decay=0.01)
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

model = train(model=net, epochs=10, loss_func=criterion, data_loader=train_loader, val_data_loader=val_loader)

torch.save(model, "./Model_tsne.pkl")