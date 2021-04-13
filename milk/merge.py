import pandas as pd
import numpy as np
import pickle

data1 = pd.read_csv("data/report.csv")

with open("combined_dict.pk", 'rb') as input_file:
    combined_dict = pickle.load(input_file)
    input_file.close()

for i in combined_dict.copy().keys():
    if i not in list(data1["乳牛編號"]):
        combined_dict.pop(i)
print(len(combined_dict))

def unique(list1):
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list


def concat(cow_id):
    ex = combined_dict[cow_id]
    ex_report = ex[0]
    ex_birth = ex[1][["分娩日期", "乾乳日期", "分娩難易度", "母牛體重"]] #以分娩日期為base
    ex_breed = ex[2][["配種日期", "配種方式", "精液種類"]] #以配種日期為base(=最後配種日期)
    ex_spec = ex[3][["乳牛編號", "狀況類別", "狀況日期"]] #以乳牛編號為base
    index = ex_report.index
    if ex_birth.values.size != 0:
        for i in range(len(list(ex_report["最近分娩日期"].values))):
            if i == 0:
                id = ex_birth[ex_birth["分娩日期"] == ex_report["最近分娩日期"].values[i]][["乾乳日期", "分娩難易度", "母牛體重"]]
                if id.values.size == 0:
                    id = pd.DataFrame(ex_birth.iloc[:1, 1:].values, columns=["乾乳日期", "分娩難易度", "母牛體重"])
            else:
                id_1 = ex_birth[ex_birth["分娩日期"] == ex_report["最近分娩日期"].values[i]][["乾乳日期", "分娩難易度", "母牛體重"]]
                if id_1.values.size == 0:
                    id_1 = pd.DataFrame(ex_birth.iloc[-1:, 1:].values, columns=["乾乳日期", "分娩難易度", "母牛體重"])
                id = pd.concat([id, id_1], axis=0)
        id.index = index
        new_report = pd.concat([ex_report, id], axis=1)
    else:
        new_report = ex_report

    ex_breed = ex_breed.append(pd.Series({"配種日期": int(0), "配種方式": int(-1), "精液種類": int(-1)}), ignore_index=True)
    for i in range(len(list(ex_report["最後配種日期"].values))):
        if i == 0:
            n_id = ex_breed[ex_breed["配種日期"] == ex_report["最後配種日期"].values[i]][["配種方式", "精液種類"]]
        else:
            n_id_1 = ex_breed[ex_breed["配種日期"] == ex_report["最後配種日期"].values[i]][["配種方式", "精液種類"]]
            n_id = pd.concat([n_id, n_id_1], axis=0)
    n_id.index = index
    new_report = pd.concat([new_report, n_id], axis=1)

    for i in range(len(list(ex_report["乳牛編號"].values))):
        if i == 0:
            m_id = pd.DataFrame(ex_spec[ex_spec["乳牛編號"] == ex_report["乳牛編號"].values[i]][["狀況類別", "狀況日期"]].values.reshape(1, -1))

        else:
            m_id_1 = pd.DataFrame(ex_spec[ex_spec["乳牛編號"] == ex_report["乳牛編號"].values[i]][["狀況類別", "狀況日期"]].values.reshape(1, -1))
            m_id = pd.concat([m_id, m_id_1], axis=0)
    m_id.index = index
    new_report = pd.concat([new_report, m_id], axis=1)
    return new_report

# example

cow_id_list = unique(data1.loc[:, "乳牛編號"].values)
cow_id = cow_id_list[317]
ex = combined_dict[cow_id]
ex_report = ex[0]
ex_birth = ex[1][["分娩日期", "乾乳日期", "分娩難易度", "母牛體重"]]  # 以分娩日期為base
ex_breed = ex[2][["配種日期", "配種方式", "精液種類"]]  # 以配種日期為base(=最後配種日期)
ex_breed = ex_breed.append(pd.Series({"配種日期": 0, "配種方式": -1, "精液種類": -1}), ignore_index=True)
print(ex_breed)
ex_spec = ex[3][["乳牛編號", "狀況類別", "狀況日期"]]
#####

for i in range(len(cow_id_list)):
    if i == 0:
        cow_data = concat(cow_id_list[i])
    else:
        cow_data = pd.concat([cow_data, concat(cow_id_list[i])], axis=0)
        print(i)

df = cow_data.copy()
df.to_csv("./data/cow_data.csv")
