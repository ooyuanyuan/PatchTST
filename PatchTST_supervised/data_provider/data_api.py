#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/21 09:54

import os
import pandas as pd

"""
自定义路径下的数据加载；
    对接的数据是 文件list, 需要一次性读取 不同阶段的数据（train，valid, predict）；
"""


def read_data_from_csv(data_path, cols, target='value', date_col='timestamp', features='S'):
    """
    read data from csv;
    :return:
    """
    raw_data = pd.read_csv(data_path)  # df_raw.columns: [date_title, ...(other features), target feature]

    if cols:  # 指定加载列，多序列的情况；
        cols = cols.copy()
        cols.remove(target)
    else:
        cols = list(raw_data.columns)
        cols.remove(target)
        cols.remove(date_col)
    raw_data = raw_data[[date_col] + cols + [target]]

    if features == "M" or features == "MS":
        cols_data = raw_data.columns[1:]
        df_data = raw_data[cols_data]
    elif features == "S":
        df_data = raw_data[[target]]
    df_data = df_data.fillna(method="ffill")  # 空值填充，是否应该按照时序索引填充， 读取之前需要先去重；
    data = df_data.values
    return raw_data, data


def select_samples(data_list_file, root_path, data_path="BCS", metric_type='ALL'):
    """
    根据指定的数据列表，获取指定目录下的数据列表；
    :param data_list_file: like as train.txt, test.txt;
    :param root_path: data root path. as: ../dataset/AIOps
    :param data_path: data_path
    :param metric_type: 'ALL' or prefix of data file name;
    :return:
    """
    data_list_file = os.path.join(root_path, data_list_file)
    data_path = os.path.join(root_path, data_path)
    # 数据保存路径，所有的数据列表；
    data_list = os.listdir(data_path) if metric_type is 'ALL' else [f for f in os.listdir(data_path) if
                                                                    metric_type in f]
    # Gets data specified in train.txt or test.txt
    cluster2csv = {}  # ={cluster: [cpu.csv]}
    for f in data_list:
        # _cluster_ = f.split("_")[0] # 集集筛选
        _cluster_ = f  # .split("_")[0] # 文件筛选
        if _cluster_ in cluster2csv:
            cluster2csv[_cluster_].append(f)
        else:
            cluster2csv[_cluster_] = [f]
    # read train/test/pred and verify that the data exists;
    with open(data_list_file, "r") as f:
        cluster_names = [line.rstrip("\n") for line in f.readlines()]
        data_samples = []
        for _cluster_ in cluster_names:
            if not _cluster_:
                continue
            data_samples.extend(cluster2csv[_cluster_])
    return data_samples


if __name__ == '__main__':
    # train_list = select_samples(data_list_file="train.txt", root_path="../dataset/AIOps", data_path='BCS')
    train_list = select_samples(data_list_file="train.txt", root_path="../../dataset/bkbase", data_path='data')
    print(len(train_list))
