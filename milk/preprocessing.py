import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from milk.Regression_model import Model, Data
from milk.Regression_model import train

data = pd.read_csv("./data/report_1.csv")
cow_data = pd.read_csv("./data/cow_data_1.csv")
data_feats = data[["資料年度", "資料月份", "酪農場代號", "胎次", "配種次數", "乳牛編號", "第一次配種精液", "最後配種精液"]].fillna(-1)
data_feats = pd.concat([data_feats, cow_data[["乾乳日期"]]], axis=1)

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

lab_feats = [unique(i) for i in data_feats.values.T]
label_data = label_featurize(data_feats.values, lab_feats)

origin_data = data[["月齡", "泌乳天數"]]
origin_data = origin_data.fillna(origin_data.mean())

latest_birth = (pd.to_datetime(data.loc[:, "最近分娩日期"]) - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, "M")
latest_mate = (pd.to_datetime(data.loc[:, "最後配種日期"]) - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, "M")

re_data = pd.concat([pd.DataFrame(label_data, columns=list(range(label_data.shape[-1]))),
                     pd.DataFrame(latest_mate), pd.DataFrame(latest_birth),
                     origin_data,
                     data[["乳量"]].fillna(-1)], axis=1)
print(re_data.shape)
print(re_data.describe())



# 3. Training
from sklearn.model_selection import train_test_split
train_data = re_data[re_data["乳量"] >= 0]
test_data = re_data[re_data["乳量"] < 0]
print("Train: ", train_data.shape, "\nTest: ", test_data.shape)

X_train = train_data.drop(columns=["乳量"])
X_train = torch.Tensor(list(X_train.values)).cuda()

y_train = train_data[["乳量"]]
y_train = torch.Tensor(list(y_train.values)).cuda()

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.001, random_state=0)

X_test = test_data.drop(columns=["乳量"])
X_test = torch.Tensor(list(X_test.values)).cuda()
torch.save(X_test, "./data/X_test_1.pkl")

print("X_train: ", X_train.shape, "y_train: ", y_train.shape, "X_val: ", X_val.shape, "X_test: ", X_test.shape)

#### Random Forest ####
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import svm
from sklearn.metrics import mean_absolute_error

#SVM = svm.SVR()
#SVM.fit(X_train.detach().cpu().numpy(), y_train.detach().cpu().numpy())
#y_pred_0 = SVM.predict(X_val.detach().cpu().numpy())
#print("SVM", mean_absolute_error(y_pred_0, y_val.detach().cpu().numpy()))

#### Predict Testing ####
#sub = pd.read_csv("./data/submission.csv")
#y_test_0 = SVM.predict(X_test.detach().cpu().numpy())
#df = pd.DataFrame(y_test_0, index=sub["ID"])
#df.to_csv("./SVM/SVM_2.csv")
###############################


# RF = RandomForestRegressor(50, random_state=0)
# RF.fit(X_train.detach().cpu().numpy(), y_train.detach().cpu().numpy())
# y_pred = RF.predict(X_val.detach().cpu().numpy())
#
# print("Random Forest regression: ", mean_absolute_error(y_pred, y_val.detach().cpu().numpy()),)
#
# #### Predict Testing ####
# sub = pd.read_csv("./data/submission.csv")
# y_test = RF.predict(X_test.detach().cpu().numpy())
# df = pd.DataFrame(y_test, index=sub["ID"])
# df.to_csv("./RF/RF_test_12.csv")
###############################

train_loader = DataLoader(dataset=Data(X_train, y_train), batch_size=128, shuffle=False)
val_loader = DataLoader(dataset=Data(X_val, y_val), batch_size=128, shuffle=False)

net = Model(n_features=X_train.shape[-1], n_hidden=200).cuda()
print(summary(net, input_size=(X_train.shape[-1],)))

criterion = nn.L1Loss().cuda()
optimizer = optim.Adam(net.parameters(), lr=.001)

model = train(model=net, epochs=50, loss_func=criterion, optimizer=optimizer, data_loader=train_loader, val_data_loader=val_loader)

torch.save(model, "./Model_22.pkl")
