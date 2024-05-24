#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/1 19:12
import os
import pathlib
import pandas as pd

"""
数据格式转换
"""
print(os.path.abspath("./"))
print(pathlib.PurePath("./", "../"))

if __name__ == '__main__':
    data_path = "/data/project/patchtst/PatchTST_supervised/dataset/bcs-v2/bcs"  # noqa
    out_path = pathlib.PurePath(data_path, "../", "bcs_data")
    os.makedirs(out_path, exist_ok=True)

    for _f in os.listdir(data_path):
        if not _f.endswith(".csv"):
            continue
        data = pd.read_csv(f"{data_path}/{_f}")
        data['timestamp'] = data['dtEventTimeStamp'] / 1000
        data['timestamp'] = data['timestamp'].astype(int)
        print(f'{data_path}/../bcs_data/{_f}')
        data[['timestamp', 'value']].to_csv(f'{out_path}/{_f}', index=False)
