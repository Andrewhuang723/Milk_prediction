import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

latest = pd.read_csv("./data/report.csv", index_col=None)

### Predict 最後配種日期
latest_1 = latest[latest["最後配種日期"].isna()]
latest_2 = latest[latest["最後配種日期"].notna()]
birth = (pd.to_datetime(latest_2.loc[:, "最近分娩日期"]) - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, "D")
birth = pd.concat([latest_2.loc[:, "泌乳天數"], birth], axis=1)
birth = birth.values.reshape(-1, 2)

mate = (pd.to_datetime(latest_2.loc[:, "最後配種日期"]) - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, "D")
mate = mate.values.reshape(-1, 1)

X_test = (pd.to_datetime(latest_1.loc[:, "最近分娩日期"]) - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, "D")
X_test = pd.concat([latest_1.loc[:, "泌乳天數"].fillna(204.89), X_test], axis=1)
X_test = X_test.values.reshape(-1, 2)

print(birth.shape, X_test.shape, mate.shape)
RF = RandomForestRegressor(100, random_state=0)
RF.fit(birth, mate)
y_train = RF.predict(birth)
y_pred = RF.predict(X_test)
y_pred = pd.Series(y_pred * np.timedelta64(1, "D") + np.datetime64('1970-01-01T00:00:00Z'))
print(mean_absolute_error(mate, y_train))

data = latest.copy()
for i in range(len(latest_1.index)):
    data.loc[latest_1.index[i], "最後配種日期"] = y_pred[i]
#data.to_csv("./data/report_1.csv")


### Predict 乾乳日期
cow_data = pd.read_csv("./data/cow_data.csv", index_col=None)
dry_milk_1 = cow_data[cow_data["乾乳日期"].isna()]
dry_milk_2 = cow_data[cow_data["乾乳日期"].notna()]

### Training datasets
birth = (pd.to_datetime(latest.loc[dry_milk_2.index, "最近分娩日期"]) - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, "D")
birth = pd.concat([latest.loc[dry_milk_2.index, "泌乳天數"].fillna(204.89), birth], axis=1)
birth = birth.values.reshape(-1, 2)

y_train = dry_milk_2.loc[:, "乾乳日期"]
y_train /= 2592000
print("std:", y_train.std())
y_train = y_train.values.reshape(-1, 1)

X_test = (pd.to_datetime(latest.loc[dry_milk_1.index, "最近分娩日期"]) - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, "D")
X_test = pd.concat([latest.loc[dry_milk_1.index, "泌乳天數"].fillna(204.89), X_test], axis=1)
print(birth.shape, y_train.shape, X_test.shape)

RF = RandomForestRegressor(100, random_state=0)
RF.fit(birth, y_train)
y_pred_0 = RF.predict(birth)
print(mean_absolute_error(y_pred=y_pred_0, y_true=y_train))
y_pred = RF.predict(X_test)

data = cow_data.copy()
for i in range(len(dry_milk_1.index)):
    data.loc[dry_milk_1.index[i], "乾乳日期"] = y_pred[i]
print(data.loc[dry_milk_1.index, "乾乳日期"])
print(data)
#data.to_csv("./data/cow_data_1.csv")