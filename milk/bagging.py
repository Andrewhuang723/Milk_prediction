import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sub = pd.read_csv("./data/submission.csv")

r = pd.read_csv("./6.44.csv").values[:, -1]
w = pd.read_csv("./6.45.csv").values[:, -1]

y_99 = pd.read_csv("./correct/y_pred_12.csv").values[:, -1]
y_11 = pd.read_csv("./correct/y_pred_01110.csv").values[:, -1]
y_15 = pd.read_csv("./correct/y_pred_15.csv").values[:, -1]
y_17 = pd.read_csv("./correct/y_pred_17.csv").values[:, -1]
y_20 = pd.read_csv("./correct/y_pred_20.csv").values[:, -1]
y_21 = pd.read_csv("./correct/y_pred_21.csv").values[:, -1]
y_22 = pd.read_csv("./correct/y_pred_22.csv").values[:, -1]
bg_18 = pd.read_csv("./bg_18.csv").values[:, -1]
rf_best = pd.read_csv("./RF/RF_test_8.csv").values[:, -1]
rf_A = pd.read_csv("./RF/RF_test_11.csv").values[:, -1]
rf_12 = pd.read_csv("./RF/RF_test_12.csv").values[:, -1]

y = (bg_18 + y_15) / 2
df = pd.DataFrame(y, index=sub["ID"])
#df.to_csv("./bg_20.csv")
y = (y_15 + y_17) / 2
df = pd.DataFrame(y, index=sub["ID"])
#df.to_csv("./bg_21.csv")
y = (y_15 + y_17 + bg_18) / 3
df = pd.DataFrame(y, index=sub["ID"])
#df.to_csv("./bg_22.csv")
y = (y_15 + y_17 + y_20 + bg_18) / 4
df = pd.DataFrame(y, index=sub["ID"])
#df.to_csv("./bg_23.csv")
y = (y_15 + y_17 + y_20 + y_21 + bg_18 + rf_best) / 6
df = pd.DataFrame(y, index=sub["ID"])
#df.to_csv("./bg_24.csv") #5.62
y = (y_15 + y_17 + y_20 + y_21 + bg_18 + rf_best + r + w) / 8
df = pd.DataFrame(y, index=sub["ID"])
#df.to_csv("./bg_25.csv")
y = (y_15 + y_17 + y_20 + y_21 + bg_18 + rf_A + rf_best) / 6
df = pd.DataFrame(y, index=sub["ID"])
#df.to_csv("./bg_26.csv") #5.62
y = (y_15 + y_17 + y_20 + y_21 + bg_18 + rf_A + rf_12 + y_22) / 8
df = pd.DataFrame(y, index=sub["ID"])
#df.to_csv("./bg_27.csv") #5.599
y = (y_15 + y_17 + y_20 + y_21 + bg_18 + rf_best + y_22) / 7
df = pd.DataFrame(y, index=sub["ID"])
#df.to_csv("./bg_28.csv") #5.609

x = ["ResNet_1", "ResNet_2", "NN_1", "NN_2", "RF"]
y = [6.44, 6.45, 5.82, 5.99, 6.26]
fig, ax = plt.subplots()
width = 0.75
ind = np.arange(len(y))
ax.barh(ind, y, width, color="y")
ax.set(xlim=[5, 6.5])
ax.set_yticks(ind+width/2)
ax.set_yticklabels(x, minor=False)
plt.title('Loss v.s Model')
plt.xlabel('Loss')
plt.ylabel('Model')
plt.show()

x = ["SVM", "RF with \nSVM features", "RF\nFirst Semen", "RF\nLast Semen", "RF\nCow ID", "RF + 3 feats"]
y = [7.470, 6.445, 6.47, 6.59, 6.41, 6.267]
fig, ax = plt.subplots(figsize=(7, 7))
width = 0.75
ind = np.arange(len(y))
ax.bar(x, y, width, color="green", )
ax.set(ylim=[5.5, 7.5])
#ax.set_yticks(ind+width/2)
plt.xticks(fontsize=7.5, rotation=45)
plt.title('Regression')
plt.ylabel('Loss')
plt.xlabel('Model & Input features')
plt.show()

x = ["Basic features", "cow ID", "Last Semen", "First Semen", "cow ID \nLast Semen", "Last Semen\nFirst Semen",
     "cow ID\nFirst Semen", "cow ID\nLast Semen\nFirst Semen"]
y = [6.42, 5.82, 6.47, 6.31, 7.27, 6.41, 6.40, 5.99]
fig, ax = plt.subplots(figsize=(7, 7))
width = 0.75
ind = np.arange(len(y))
ax.bar(x, y, width, color="green", )
ax.set(ylim=[5.5, 7.5])
#ax.set_yticks(ind+width/2)
plt.xticks(fontsize=7.5, rotation=45)
plt.title('NN model')
plt.ylabel('Loss')
plt.xlabel('Input features')
plt.show()

x = ["NN model 1\n10 epochs", "NN model 1\n50 epochs", "NN model 2\n10 epochs", "NN model 2\n50 epochs"]
y = [5.82, 5.89, 5.99, 6.26]
fig, ax = plt.subplots(figsize=(7,7))
width = 0.85
ind = np.arange(len(y))
ax.bar([1, 2, 3.5, 4.5], y, width, color=["gray", "brown", "gray", "brown"] )
ax.set(ylim=[5.5, 6.5])
#ax.set_yticks(ind+width/2)
plt.xticks([1, 2, 3.5, 4.5], x, fontsize=7.5, )
plt.title('Loss v.s Epochs')
plt.ylabel('Loss')
plt.xlabel('Model with different epochs')
plt.show()