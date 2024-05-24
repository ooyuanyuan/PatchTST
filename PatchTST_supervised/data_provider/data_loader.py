import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')
TZ = "Asia/Shanghai"


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]  # 336=14D
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:  # 时间编码, 时间相关特征；
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)  # .T

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        # 边界处理：每类数据的长度+频率不同；
        # 24*60=> 24天，24+8天, 24+8+8
        #  0 -- 24 -- 24+8
        #       24 -- 24+8 -- 24+16
        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)  # 7:2:1的比例
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test

        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    # 预测时支持单文件预测；
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols  # 预测时指定；
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)  # 这里的确有点：数据分布不一致时；
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dattest_AIOps_Custom(Dataset):  # noqa
    """
    自定义数据解析：目前支持数据类型：
    配置各阶段的数据列表(train.txt, test.txt) + 数据目录
    """

    def __init__(
            self, root_path, data_path, pretrain_data_list, test_data_list, valid_data_list,
            size, flag='train', features='S', target='value',
            timeenc=0, freq='t', cols=None, date_col='timestamp', metric_type='ALL'
    ):
        """
        :param root_path:
        :param data_path:  none;
        :param pretrain_data_list:  like train.txt, test.txt
        :param test_data_list:  test.txt
        :param valid_data_list:  like valid.txt
        :param size:
        :param flag: ['train', 'test', 'val']
        :param features:
        :param target:
        :param timeenc:
        :param freq:
        :param cols:
        :param metric_type:  # metric type like ['cpu', 'mem', 'ALL']
        :param date_col:
        """
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.timeenc = timeenc
        self.freq = freq
        self.date_col = date_col
        self.cols = cols
        self.metric_type = metric_type

        self.root_path = root_path
        self.data_path = data_path
        self.pretrain_data_list = pretrain_data_list  # train.txt;
        self.test_data_list = test_data_list  # train.txt;
        self.valid_data_list = valid_data_list  # train.txt;
        self.__read_data__()

    def __read_data__(self):
        """
        批量处理所有的csv文件中的时序数据，转成训练的一条序列样本，并处理成批训练样本（DataLoader）
        :return:
        """
        from data_provider.data_api import select_samples

        assert self.set_type in [0, 1, 2]  # {'train': 0, 'val': 1, 'test': 2}
        self.data_list_file = self.pretrain_data_list if self.set_type == 0 else (
            self.test_data_list if self.set_type == 2 else self.valid_data_list)  # train.txt, test.
        # "BCS"
        data_list = select_samples(self.data_list_file, root_path=self.root_path, data_path=self.data_path,
                                   metric_type=self.metric_type)
        # 全局标准化在这里实现更好；
        self.seq_x, self.seq_y = [], []
        self.seq_x_mark, self.seq_y_mark = [], []

        # 避免序列连接的断崖问题
        for _f in data_list:
            data, data_stamp = self.__read_data_from_csv(os.path.join(self.root_path, self.data_path, _f))
            for s_begin in range(len(data) - self.seq_len - self.pred_len + 1):
                s_end = s_begin + self.seq_len  # 序列长度；

                r_begin = s_end - self.label_len
                r_end = r_begin + self.label_len + self.pred_len

                seq_x = data[s_begin:s_end]
                seq_y = data[r_begin:r_end]  # X=用前14天的数据， y=后7天数据
                seq_x_mark = data_stamp[s_begin:s_end]
                seq_y_mark = data_stamp[r_begin:r_end]
                self.seq_x.append(seq_x)
                self.seq_y.append(seq_y)

                self.seq_x_mark.append(seq_x_mark)
                self.seq_y_mark.append(seq_y_mark)

    def __getitem__(self, index):
        return self.seq_x[index], self.seq_y[index], self.seq_x_mark[index], self.seq_y_mark[index]

    def __len__(self):
        # return len(self.data_x) - self.seq_len - self.pred_len + 1
        return len(self.seq_x)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def __read_data_from_csv(self, data_path):
        """
        具体获取数据的实现接口： 读取CSV文件时序，处理：
            1、归一化
            2、时间特征编码；
        :return:
        """
        # default StandardScaler;
        self.scaler = StandardScaler()

        df_raw = pd.read_csv(data_path)  # df_raw.columns: [date_title, ...(other features), target feature]
        # preprocess: drop duplicate and sorted
        df_raw = df_raw.drop_duplicates(self.date_col, keep='last').sort_values(by=self.date_col)
        df_raw = df_raw.fillna(method="ffill")  # 空值填充，是否应该按照时序索引填充， 读取之前需要先去重；

        if self.cols:  # 指定加载列，多序列的情况；
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove(self.date_col)
        df_raw = df_raw[[self.date_col] + cols + [self.target]]  # sorted cols
        # get features and scale;
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        '''
        # 数据预处理标准化：
        #   - 全局标准化，所有的训练数据一起标准化；
        #   - 局部标准化，单个样本标准化；目前：这种方式
        '''
        if self.scaler:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        # timestamp process
        df_stamp = df_raw[[self.date_col]]
        df_stamp[self.date_col] = pd.to_datetime(list(df_stamp[self.date_col]), unit='s', utc=True).tz_convert(
            TZ).strftime("%Y-%m-%d %H:%M:%S")

        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop([self.date_col], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp[self.date_col].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        return data, data_stamp


class Dattest_AIOps_Pred(Dataset):
    # 预测时支持单文件预测；
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None, date_col='timestamp',
                 return_y_stamp=True, metric_type='ALL', **kwargs
                 ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.inverse = inverse
        self.freq = freq

        self.return_y_stamp = return_y_stamp  # return y timestamp and plot or viewer;
        self.date_col = date_col
        self.cols = cols
        self.metric_type = metric_type

        self.root_path = root_path
        self.data_path = data_path
        self.test_data_list = kwargs.get("test_data_list", 'test.csv')
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        from data_provider.data_api import select_samples
        # single file test;
        data_list = select_samples(self.test_data_list, root_path=self.root_path, data_path=self.data_path,
                                   metric_type=self.metric_type)
        self.test_csv = data_list[0]
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path, data_list[0]))
        '''
        df_raw.columns: ['timestamp', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)  # 预测的特征列；
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove(self.date_col)

        df_raw = df_raw[[self.date_col] + cols + [self.target]]

        # border1 = len(df_raw) - self.seq_len  # seq_len：预测时或提取特征时依赖的数据历史点；
        # border1 = self.seq_len
        # border2 = len(df_raw)
        border1 = 0
        border2 = len(df_raw)  # - self.seq_len
        print(f"predict index: {border1}-{border2}")

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)  # 这里的确有点点：数据分布不一致时；
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[[self.date_col]][border1:border2]
        df_stamp[self.date_col] = pd.to_datetime(list(df_stamp[self.date_col]), unit='s', utc=True).tz_convert(
            TZ).strftime("%Y-%m-%d %H:%M:%S")

        # not process;
        # pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)  #
        # df_stamp = pd.DataFrame(columns=['date'])
        # df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            # df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            # df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop([self.date_col], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp[self.date_col].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]  # data=fit-transformer>value
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp  # 时间编码特征
        self.sour_stamp = df_raw[[self.date_col]]  # [border1:border2]  # source datetime;

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len  # 0-2880
        r_begin = s_end - self.label_len  # 2880-2880+1440
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_end]
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.return_y_stamp:
            seq_y_stamp = np.array(
                self.sour_stamp[r_begin:r_end].values.astype(int)
            )
            return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_y_stamp
        else:
            return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1 - self.pred_len  # 最后一个窗口不用再训练；

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dattest_AIOps_Pred_2(Dataset):
    """
    预测时支持单文件预测；
    """

    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None, date_col='timestamp',
                 return_y_stamp=True, metric_type='ALL', **kwargs
                 ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.inverse = inverse
        self.freq = freq

        self.return_y_stamp = return_y_stamp  # return y timestamp and plot or viewer;
        self.date_col = date_col
        self.cols = cols
        self.metric_type = metric_type

        self.root_path = root_path
        self.data_path = data_path
        self.test_data_list = kwargs.get("test_data_list", 'test.csv')
        self.test_data = kwargs.get("test_data")  # 特有数据；
        self.__read_data__()

    def __read_data__(self):
        """迁移学习，进行预测"""

        self.scaler = StandardScaler()
        # single file test;
        print(self.test_data)
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path, self.test_data))
        """
        df_raw.columns: ['timestamp', ...(other features), target feature]
        """
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)  # 预测的特征列；
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove(self.date_col)

        df_raw = df_raw[[self.date_col] + cols + [self.target]]

        # border1 = len(df_raw) - self.seq_len  # seq_len：预测时或提取特征时依赖的数据历史点；
        # border1 = self.seq_len
        # border2 = len(df_raw)
        border1 = 0
        border2 = len(df_raw)  # - self.seq_len
        print(f"predict index: {border1}-{border2}")

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)  # 这里的确有点小问题，验证过大部署不是问题：数据分布不一致时；
            data = self.scaler.transform(df_data.values)  # 标准化后的结果；
        else:
            data = df_data.values

        df_stamp = df_raw[[self.date_col]][border1:border2]
        df_stamp[self.date_col] = pd.to_datetime(list(df_stamp[self.date_col]), unit='s', utc=True).tz_convert(
            TZ).strftime("%Y-%m-%d %H:%M:%S")

        # not process;
        # pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)  #
        # df_stamp = pd.DataFrame(columns=['date'])
        # df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            # df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            # df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop([self.date_col], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp[self.date_col].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]  # data=fit-transformer>value
        if self.inverse:  #
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp  # 时间编码特征
        self.sour_stamp = df_raw[[self.date_col]]  # [border1:border2]  # source datetime;

    def __getitem__(self, index):  # 实现__getitem__和 __len__方法，就会被认为是序列；可以像序列一样访问；
        s_begin = index
        s_end = s_begin + self.seq_len  # 0-2880
        r_begin = s_end - self.label_len  # 2880-2880+1440
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_end]
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.return_y_stamp:
            seq_y_stamp = np.array(
                self.sour_stamp[r_begin:r_end].values.astype(int)
            )
            return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_y_stamp
        else:
            return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1 - self.pred_len  # 最后一个窗口不用再训练；

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
