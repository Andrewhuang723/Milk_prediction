import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import os
from milk.Regression_model import Model, Data
import pandas as pd

os.environ['KMP_DUPLICATE_LIB_OK']='True'
device = torch.device("cuda")

dict = torch.load("./Model_22.pkl")
loss = dict["loss"]
val_loss = dict["val_loss"]
loss = torch.stack(loss)
val_loss = torch.stack(val_loss)
plt.figure()
plt.plot(range(len(loss)), loss.detach().cpu().numpy(), label="Training")
plt.plot(range(len(val_loss)), val_loss.detach().cpu().numpy(), label="Validation")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()

X_test = torch.load("./data/X_test_1.pkl")
sub = pd.read_csv("data/submission.csv")
del dict["loss"]
del dict["val_loss"]
model = Model(X_test.shape[-1], 200).cuda()
model.load_state_dict(dict)
model.eval()
y_pred = model(X_test)

y_pred = y_pred.detach().cpu().numpy()
df = pd.DataFrame(y_pred, index=sub["ID"])
df.to_csv("./correct/y_pred_22.csv")

import numpy as np
from sklearn.metrics import mean_absolute_error
correct = pd.read_csv("bg_9.csv").values[:, -1]
print(mean_absolute_error(y_true=correct, y_pred=y_pred))
