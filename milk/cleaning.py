import datetime
import itertools
import math
import matplotlib.pyplot as pyplot
from multiprocessing import Pool
import numpy
import os
import pandas
import pickle
import random
import re
from sklearn.preprocessing import LabelEncoder
import time
import torch
from torch import optim
import zlib


########## Classes ##########
class CONST:
    report_col = lambda: ['ID', '資料年度', '資料月份', '酪農場代號',
                          '乳牛編號', '父親牛精液編號', '母親乳牛編號',
                          '出生日期', '胎次', '泌乳天數', '乳量',
                          '最近分娩日期', '採樣日期', '月齡', '檢測日期',
                          '最後配種日期', '最後配種精液', '配種次數',
                          '前次分娩日期', '第一次配種日期', '第一次配種精液']
    birth_col = lambda: ['乳牛編號', '分娩日期', '乾乳日期', '犢牛編號1',
                         '犢牛編號2', '母牛體重', '登錄日期', '計算胎次',
                         '胎次', '分娩難易度', '犢牛體型', '犢牛性別',
                         '酪農場代號']
    breed_col = lambda: ['乳牛編號', '配種日期', '配種精液', '登錄日期',
                         '孕檢', '配種方式', '精液種類', '酪農場代號']
    spec_col = lambda: ['乳牛編號', '狀況類別', '狀況代號', '狀況日期',
                        '備註', '登錄日期', '酪農場代號']
    dupl_col = lambda: ['乳牛編號', '胎次', '登錄日期', '酪農場代號']
    unix_time0 = lambda: datetime.datetime(1970, 1, 1)


########## Functions ##########
def readCSV(file_path):
    fp = open(file_path, 'r', encoding='utf-8')

    if 'report' in file_path:
        input_df = pandas.read_csv(fp, header=None, skiprows=[0], low_memory=False, names=CONST.report_col())
        # Convert date-like string into unix epoch timestamp
        col_list = ['出生日期', '最近分娩日期', '採樣日期', '檢測日期', '最後配種日期', '前次分娩日期', '第一次配種日期']
        for col_name in col_list:
            input_df[col_name] = input_df[col_name].apply(str2UnixEpoch)
    elif 'birth' in file_path:
        input_df = pandas.read_csv(fp, header=None, skiprows=[0], low_memory=False, names=CONST.birth_col())
        # Convert date-like string into unix epoch timestamp
        col_list = ['分娩日期', '乾乳日期', '登錄日期']
        for col_name in col_list:
            input_df[col_name] = input_df[col_name].apply(str2UnixEpoch)
    elif 'breed' in file_path:
        input_df = pandas.read_csv(fp, header=None, skiprows=[0], low_memory=False, names=CONST.breed_col())
        # Convert date-like string into unix epoch timestamp
        col_list = ['配種日期', '登錄日期']
        for col_name in col_list:
            input_df[col_name] = input_df[col_name].apply(str2UnixEpoch)
    elif 'spec' in file_path:
        input_df = pandas.read_csv(fp, header=None, skiprows=[0], low_memory=False, names=CONST.spec_col())
        # Convert date-like string into unix epoch timestamp
        col_list = ['狀況日期', '登錄日期']
        for col_name in col_list:
            input_df[col_name] = input_df[col_name].apply(str2UnixEpoch)
    else:
        print("Try to open an unknown .csv file.")
        input_df = None
        fp.close()
        exit()

    fp.close()
    return input_df


def str2UnixEpoch(date_str):
    date_unix = 0

    if type(date_str) == str:
        match = re.match(r'(?P<y>\d+)\/(?P<m>\d+)\/(?P<d>\d+)\s+(?P<hour>\d+)\:(?P<minute>\d+)', date_str)
        date_datetime = datetime.datetime(year=int(match.group('y')),
                                          month=int(match.group('m')),
                                          day=int(match.group('d')),
                                          hour=int(match.group('hour')),
                                          minute=int(match.group('minute')))
        date_unix = int((date_datetime - CONST.unix_time0()).total_seconds())
    else:
        date_unix = 0

    return date_unix


########## Main function ##########
if __name__ == '__main__':
    # Display pandas DataFrame without truncation
    pandas.set_option('display.max_columns', None)
    pandas.set_option('display.max_rows', None)

    # Read .csv
    report_df = readCSV('data/report.csv')
    birth_df = readCSV('data/birth.csv')
    breed_df = readCSV('data/breed.csv')
    spec_df = readCSV('data/spec.csv')

    # Merge all DFs
    combined_columns = CONST.report_col() + ['birth'] + ['breed'] + ['spec']
    cow_id = 87121677

    # (1) report_df
    big_df = pandas.DataFrame(columns=combined_columns)
    big_df = pandas.concat([big_df, report_df])

    print(report_df.loc[report_df['乳牛編號'] == cow_id])
    print("#############################################")
    print(birth_df.loc[birth_df['乳牛編號'] == cow_id])
    print("#############################################")
    print(breed_df.loc[breed_df['乳牛編號'] == cow_id])
    print("#############################################")
    print(spec_df.loc[spec_df['乳牛編號'] == cow_id])

    combined_dict = {}
    all_cow_id = set(report_df['乳牛編號']).union(set(birth_df['乳牛編號']), set(breed_df['乳牛編號']), set(spec_df['乳牛編號']))

    for cow_id in all_cow_id:
        combined_dict[cow_id] = (report_df.loc[report_df['乳牛編號'] == cow_id],
                                 birth_df.loc[birth_df['乳牛編號'] == cow_id],
                                 breed_df.loc[breed_df['乳牛編號'] == cow_id],
                                 spec_df.loc[spec_df['乳牛編號'] == cow_id])

    with open('combined_dict.pk', 'wb') as f:
        pickle.dump(combined_dict, f)
        f.close()
'''
combined_dict = {cow_id : (report_DataFrame, birth_DataFrame, breed_DataFrame, spec_DataFrame)}
                           └─ the slice of report DataFrame to specific cow id
'''
