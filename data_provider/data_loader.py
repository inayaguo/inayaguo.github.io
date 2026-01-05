import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
# from data_provider.m4 import M4Dataset, M4Meta
# from data_provider.uea import subsample, interpolate_missing, Normalizer
# from sktime.datasets import load_from_tsfile_to_dataframe
import warnings

warnings.filterwarnings('ignore')


def select_train_test(df):
        all_groups = []

        grouped = df.groupby(['name', 'start'])

        for name_start, group in grouped:
                if len(group) < 13:  # 如果组内数据不足 13 行，则跳过
                    continue
                
                # 计算前 12 行的均值
                mean_value = group.iloc[:12]['predict'].mean()  # 假设 'value' 是需要计算的列
                # 取第 13 行的值
                thirteenth_value = group.iloc[12]['predict']
                
                # 计算差异率
                if mean_value != 0:  # 避免除以零的情况
                    difference_rate = abs(thirteenth_value - mean_value) / mean_value
                    
                    # 如果差异率小于等于 10%，则保留该组
                    if difference_rate <= 0.30:
                        all_groups.append(group)

        result_df = pd.concat(all_groups, ignore_index=True)

        return result_df

# 销量预测
class Sale_Prediction(Dataset):
    def __init__(self, root_path, args, flag='train', size=None,
                 features='S', data_path='sale.csv',
                 target='OT', scale=False, timeenc=0, freq='h', train_only=False):
        self.args = args
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
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
        global df_data
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        
        print('before', df_raw.shape)
        
        '''数据清洗'''
        df_raw = select_train_test(df_raw)
        print('after', df_raw.shape)

        df_raw = df_raw.fillna(100)

        # 选择需要的特征
        group_columns = ['start', 'name']
        stamp_columns = ['month']
        final_columns = ['predict']  # 基本特征字段
        slide_columns = ['predict_3', 'predict_2', 'predict_1', 'predict_15', 'predict_14', 'predict_13', 'predict_12']
        trends_columns = ['mean', 'mean_past', 'standard',
                          'standard_past', 'predict_3_2', 'predict_2_1', 'predict_3_2_past',
                          'predict_2_1_past', 'trend_mean', 'trend_mean_past', 'change_1',
                          'change_2', 'change_3']
        # 'mean_1', 'id', 'name', 'mean'
        # 选取不同的输入特征
        if 'slide' in self.features:
            final_columns += slide_columns
        if 'trends' in self.features:
            final_columns += trends_columns

        def extract_number(s):
            return int(s.split('month')[-1])


        df_raw = df_raw[final_columns + group_columns+stamp_columns]
        # df_raw['predict_mean'] = df_raw.groupby(['start', 'name'])['predict'].transform('mean')  
        # df_raw = df_raw[df_raw['predict_mean'] > 35000]

        # 划分训练集和测试集
        month_value = sorted(df_raw['start'].unique())
        train_months, test_months = month_value[:-1], month_value[-1]
        df_train = df_raw[df_raw['start'].isin(train_months)]
        df_test = df_raw[df_raw['start'] == test_months]

        # 划分训练集和测试集
        # month_value = self.args.month_predict
        ## train_months, test_months = ['month'+str(month_value-4), 'month'+str(month_value-3), 'month'+str(month_value-2), 'month'+str(month_value-1)], 'month'+str(month_value)
        # train_months, test_months = [month_value-4, month_value-3, month_value-2, month_value-1], month_value
        # df_train = df_raw[df_raw['start'].isin(train_months)]
        # df_test = df_raw[df_raw['start'] == test_months]

        # 根据任务类型选取使用的数据集
        df = df_train if self.set_type == 0 else df_test
        # 划分输入和输出
        grouped = df.groupby(['start', 'name'])
        # grouped = df.groupby(['start', 'id'])
        group_len = len(grouped)
        data_selected = np.empty((group_len, self.seq_len + self.pred_len, len(final_columns)))
        stamp_length = 2 if self.timeenc == 0 else 1
        data_stamp = np.empty((group_len, self.seq_len + self.pred_len, stamp_length))
        for index, group_pack in enumerate(grouped):
            group = group_pack[1]
            # 处理非时间特征数据
            group_features = group[final_columns]
            if group_features.shape[0] != 13:
                continue
            data_selected[index] = group_features.values
            # 处理时间特征数据，后续进行temporal_embedding
            group_stamp = group[stamp_columns]
            # group_stamp['date'] = pd.to_datetime(group_stamp.month,format='%Y%m')
            group_stamp['date'] = group_stamp.month
            if self.timeenc == 0:
                group_stamp['month'] = group_stamp.date.apply(lambda row: row.month, 1)
                group_stamp['year'] = group_stamp.date.apply(lambda row: row.year, 1)
                group_stamp = group_stamp.drop(['date'], 1).values
            else:
                # group_stamp = time_features(pd.to_datetime(group_stamp['date'].values), freq=self.freq)
                group_stamp = group_stamp['date']
                # .apply(extract_number)
                # group_stamp = group_stamp.transpose(1, 0)
            # data_stamp[index] = group_stamp
            data_stamp[index] = group_stamp.values.reshape(-1, 1)

        print('input_shape', data_stamp.shape)
        self.data = data_selected
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        sample = self.data[index]
        sample_stamp = self.data_stamp[index]
        s_begin = 0
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = sample[s_begin:s_end]
        seq_y = sample[r_begin:r_end]
        seq_x_mark = sample_stamp[s_begin:s_end]
        seq_y_mark = sample_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
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

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
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
            data_stamp = df_stamp.drop(['date'], 1).values
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

# class Dataset_ETT_minute(Dataset):
#     def __init__(self, root_path, flag='train', size=None,
#                  features='S', data_path='ETTm1.csv',
#                  target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
#         # size [seq_len, label_len, pred_len]
#         # info
#         if size == None:
#             self.seq_len = 24 * 4 * 4
#             self.label_len = 24 * 4
#             self.pred_len = 24 * 4
#         else:
#             self.seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]
#         # init
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.set_type = type_map[flag]

#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.timeenc = timeenc
#         self.freq = freq

#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()

#     def __read_data__(self):
#         self.scaler = StandardScaler()
#         df_raw = pd.read_csv(os.path.join(self.root_path,
#                                           self.data_path))

#         border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
#         border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
#         border1 = border1s[self.set_type]
#         border2 = border2s[self.set_type]

#         if self.features == 'M' or self.features == 'MS':
#             cols_data = df_raw.columns[1:]
#             df_data = df_raw[cols_data]
#         elif self.features == 'S':
#             df_data = df_raw[[self.target]]

#         if self.scale:
#             train_data = df_data[border1s[0]:border2s[0]]
#             self.scaler.fit(train_data.values)
#             data = self.scaler.transform(df_data.values)
#         else:
#             data = df_data.values

#         df_stamp = df_raw[['date']][border1:border2]
#         df_stamp['date'] = pd.to_datetime(df_stamp.date)
#         if self.timeenc == 0:
#             df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
#             df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
#             df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
#             df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
#             df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
#             df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
#             data_stamp = df_stamp.drop(['date'], 1).values
#         elif self.timeenc == 1:
#             data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
#             data_stamp = data_stamp.transpose(1, 0)

#         self.data_x = data[border1:border2]
#         self.data_y = data[border1:border2]
#         self.data_stamp = data_stamp

#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len

#         seq_x = self.data_x[s_begin:s_end]
#         seq_y = self.data_y[r_begin:r_end]
#         seq_x_mark = self.data_stamp[s_begin:s_end]
#         seq_y_mark = self.data_stamp[r_begin:r_end]

#         return seq_x, seq_y, seq_x_mark, seq_y_mark

#     def __len__(self):
#         return len(self.data_x) - self.seq_len - self.pred_len + 1

#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)


# class Dataset_Custom(Dataset):
#     def __init__(self, root_path, flag='train', size=None,
#                  features='S', data_path='ETTh1.csv',
#                  target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
#         # size [seq_len, label_len, pred_len]
#         # info
#         if size == None:
#             self.seq_len = 24 * 4 * 4
#             self.label_len = 24 * 4
#             self.pred_len = 24 * 4
#         else:
#             self.seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]
#         # init
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.set_type = type_map[flag]

#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.timeenc = timeenc
#         self.freq = freq

#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()

#     def __read_data__(self):
#         self.scaler = StandardScaler()
#         df_raw = pd.read_csv(os.path.join(self.root_path,
#                                           self.data_path))

#         '''
#         df_raw.columns: ['date', ...(other features), target feature]
#         '''
#         cols = list(df_raw.columns)
#         cols.remove(self.target)
#         cols.remove('date')
#         df_raw = df_raw[['date'] + cols + [self.target]]
#         num_train = int(len(df_raw) * 0.7)
#         num_test = int(len(df_raw) * 0.2)
#         num_vali = len(df_raw) - num_train - num_test
#         border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
#         border2s = [num_train, num_train + num_vali, len(df_raw)]
#         border1 = border1s[self.set_type]
#         border2 = border2s[self.set_type]

#         if self.features == 'M' or self.features == 'MS':
#             cols_data = df_raw.columns[1:]
#             df_data = df_raw[cols_data]
#         elif self.features == 'S':
#             df_data = df_raw[[self.target]]

#         if self.scale:
#             train_data = df_data[border1s[0]:border2s[0]]
#             self.scaler.fit(train_data.values)
#             data = self.scaler.transform(df_data.values)
#         else:
#             data = df_data.values

#         df_stamp = df_raw[['date']][border1:border2]
#         df_stamp['date'] = pd.to_datetime(df_stamp.date)
#         if self.timeenc == 0:
#             df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
#             df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
#             df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
#             df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
#             data_stamp = df_stamp.drop(['date'], 1).values
#         elif self.timeenc == 1:
#             data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
#             data_stamp = data_stamp.transpose(1, 0)

#         self.data_x = data[border1:border2]
#         self.data_y = data[border1:border2]
#         self.data_stamp = data_stamp

#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len

#         seq_x = self.data_x[s_begin:s_end]
#         seq_y = self.data_y[r_begin:r_end]
#         seq_x_mark = self.data_stamp[s_begin:s_end]
#         seq_y_mark = self.data_stamp[r_begin:r_end]

#         return seq_x, seq_y, seq_x_mark, seq_y_mark

#     def __len__(self):
#         return len(self.data_x) - self.seq_len - self.pred_len + 1

#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)


# class Dataset_M4(Dataset):
#     def __init__(self, root_path, flag='pred', size=None,
#                  features='S', data_path='ETTh1.csv',
#                  target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
#                  seasonal_patterns='Yearly'):
#         # size [seq_len, label_len, pred_len]
#         # init
#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.inverse = inverse
#         self.timeenc = timeenc
#         self.root_path = root_path

#         self.seq_len = size[0]
#         self.label_len = size[1]
#         self.pred_len = size[2]

#         self.seasonal_patterns = seasonal_patterns
#         self.history_size = M4Meta.history_size[seasonal_patterns]
#         self.window_sampling_limit = int(self.history_size * self.pred_len)
#         self.flag = flag

#         self.__read_data__()

#     def __read_data__(self):
#         # M4Dataset.initialize()
#         if self.flag == 'train':
#             dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
#         else:
#             dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
#         training_values = np.array(
#             [v[~np.isnan(v)] for v in
#              dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
#         self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
#         self.timeseries = [ts for ts in training_values]

#     def __getitem__(self, index):
#         insample = np.zeros((self.seq_len, 1))
#         insample_mask = np.zeros((self.seq_len, 1))
#         outsample = np.zeros((self.pred_len + self.label_len, 1))
#         outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

#         sampled_timeseries = self.timeseries[index]
#         cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
#                                       high=len(sampled_timeseries),
#                                       size=1)[0]

#         insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
#         insample[-len(insample_window):, 0] = insample_window
#         insample_mask[-len(insample_window):, 0] = 1.0
#         outsample_window = sampled_timeseries[
#                            cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
#         outsample[:len(outsample_window), 0] = outsample_window
#         outsample_mask[:len(outsample_window), 0] = 1.0
#         return insample, outsample, insample_mask, outsample_mask

#     def __len__(self):
#         return len(self.timeseries)

#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)

#     def last_insample_window(self):
#         """
#         The last window of insample size of all timeseries.
#         This function does not support batching and does not reshuffle timeseries.

#         :return: Last insample window of all timeseries. Shape "timeseries, insample size"
#         """
#         insample = np.zeros((len(self.timeseries), self.seq_len))
#         insample_mask = np.zeros((len(self.timeseries), self.seq_len))
#         for i, ts in enumerate(self.timeseries):
#             ts_last_window = ts[-self.seq_len:]
#             insample[i, -len(ts):] = ts_last_window
#             insample_mask[i, -len(ts):] = 1.0
#         return insample, insample_mask


# class PSMSegLoader(Dataset):
#     def __init__(self, root_path, win_size, step=1, flag="train"):
#         self.flag = flag
#         self.step = step
#         self.win_size = win_size
#         self.scaler = StandardScaler()
#         data = pd.read_csv(os.path.join(root_path, 'train.csv'))
#         data = data.values[:, 1:]
#         data = np.nan_to_num(data)
#         self.scaler.fit(data)
#         data = self.scaler.transform(data)
#         test_data = pd.read_csv(os.path.join(root_path, 'test.csv'))
#         test_data = test_data.values[:, 1:]
#         test_data = np.nan_to_num(test_data)
#         self.test = self.scaler.transform(test_data)
#         self.train = data
#         self.val = self.test
#         self.test_labels = pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]
#         print("test:", self.test.shape)
#         print("train:", self.train.shape)

#     def __len__(self):
#         if self.flag == "train":
#             return (self.train.shape[0] - self.win_size) // self.step + 1
#         elif (self.flag == 'val'):
#             return (self.val.shape[0] - self.win_size) // self.step + 1
#         elif (self.flag == 'test'):
#             return (self.test.shape[0] - self.win_size) // self.step + 1
#         else:
#             return (self.test.shape[0] - self.win_size) // self.win_size + 1

#     def __getitem__(self, index):
#         index = index * self.step
#         if self.flag == "train":
#             return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
#         elif (self.flag == 'val'):
#             return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
#         elif (self.flag == 'test'):
#             return np.float32(self.test[index:index + self.win_size]), np.float32(
#                 self.test_labels[index:index + self.win_size])
#         else:
#             return np.float32(self.test[
#                               index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
#                 self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


# class MSLSegLoader(Dataset):
#     def __init__(self, root_path, win_size, step=1, flag="train"):
#         self.flag = flag
#         self.step = step
#         self.win_size = win_size
#         self.scaler = StandardScaler()
#         data = np.load(os.path.join(root_path, "MSL_train.npy"))
#         self.scaler.fit(data)
#         data = self.scaler.transform(data)
#         test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
#         self.test = self.scaler.transform(test_data)
#         self.train = data
#         self.val = self.test
#         self.test_labels = np.load(os.path.join(root_path, "MSL_test_label.npy"))
#         print("test:", self.test.shape)
#         print("train:", self.train.shape)

#     def __len__(self):
#         if self.flag == "train":
#             return (self.train.shape[0] - self.win_size) // self.step + 1
#         elif (self.flag == 'val'):
#             return (self.val.shape[0] - self.win_size) // self.step + 1
#         elif (self.flag == 'test'):
#             return (self.test.shape[0] - self.win_size) // self.step + 1
#         else:
#             return (self.test.shape[0] - self.win_size) // self.win_size + 1

#     def __getitem__(self, index):
#         index = index * self.step
#         if self.flag == "train":
#             return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
#         elif (self.flag == 'val'):
#             return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
#         elif (self.flag == 'test'):
#             return np.float32(self.test[index:index + self.win_size]), np.float32(
#                 self.test_labels[index:index + self.win_size])
#         else:
#             return np.float32(self.test[
#                               index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
#                 self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


# class SMAPSegLoader(Dataset):
#     def __init__(self, root_path, win_size, step=1, flag="train"):
#         self.flag = flag
#         self.step = step
#         self.win_size = win_size
#         self.scaler = StandardScaler()
#         data = np.load(os.path.join(root_path, "SMAP_train.npy"))
#         self.scaler.fit(data)
#         data = self.scaler.transform(data)
#         test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
#         self.test = self.scaler.transform(test_data)
#         self.train = data
#         self.val = self.test
#         self.test_labels = np.load(os.path.join(root_path, "SMAP_test_label.npy"))
#         print("test:", self.test.shape)
#         print("train:", self.train.shape)

#     def __len__(self):

#         if self.flag == "train":
#             return (self.train.shape[0] - self.win_size) // self.step + 1
#         elif (self.flag == 'val'):
#             return (self.val.shape[0] - self.win_size) // self.step + 1
#         elif (self.flag == 'test'):
#             return (self.test.shape[0] - self.win_size) // self.step + 1
#         else:
#             return (self.test.shape[0] - self.win_size) // self.win_size + 1

#     def __getitem__(self, index):
#         index = index * self.step
#         if self.flag == "train":
#             return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
#         elif (self.flag == 'val'):
#             return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
#         elif (self.flag == 'test'):
#             return np.float32(self.test[index:index + self.win_size]), np.float32(
#                 self.test_labels[index:index + self.win_size])
#         else:
#             return np.float32(self.test[
#                               index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
#                 self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


# class SMDSegLoader(Dataset):
#     def __init__(self, root_path, win_size, step=100, flag="train"):
#         self.flag = flag
#         self.step = step
#         self.win_size = win_size
#         self.scaler = StandardScaler()
#         data = np.load(os.path.join(root_path, "SMD_train.npy"))
#         self.scaler.fit(data)
#         data = self.scaler.transform(data)
#         test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
#         self.test = self.scaler.transform(test_data)
#         self.train = data
#         data_len = len(self.train)
#         self.val = self.train[(int)(data_len * 0.8):]
#         self.test_labels = np.load(os.path.join(root_path, "SMD_test_label.npy"))

#     def __len__(self):
#         if self.flag == "train":
#             return (self.train.shape[0] - self.win_size) // self.step + 1
#         elif (self.flag == 'val'):
#             return (self.val.shape[0] - self.win_size) // self.step + 1
#         elif (self.flag == 'test'):
#             return (self.test.shape[0] - self.win_size) // self.step + 1
#         else:
#             return (self.test.shape[0] - self.win_size) // self.win_size + 1

#     def __getitem__(self, index):
#         index = index * self.step
#         if self.flag == "train":
#             return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
#         elif (self.flag == 'val'):
#             return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
#         elif (self.flag == 'test'):
#             return np.float32(self.test[index:index + self.win_size]), np.float32(
#                 self.test_labels[index:index + self.win_size])
#         else:
#             return np.float32(self.test[
#                               index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
#                 self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


# class SWATSegLoader(Dataset):
#     def __init__(self, root_path, win_size, step=1, flag="train"):
#         self.flag = flag
#         self.step = step
#         self.win_size = win_size
#         self.scaler = StandardScaler()

#         train_data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
#         test_data = pd.read_csv(os.path.join(root_path, 'swat2.csv'))
#         labels = test_data.values[:, -1:]
#         train_data = train_data.values[:, :-1]
#         test_data = test_data.values[:, :-1]

#         self.scaler.fit(train_data)
#         train_data = self.scaler.transform(train_data)
#         test_data = self.scaler.transform(test_data)
#         self.train = train_data
#         self.test = test_data
#         self.val = test_data
#         self.test_labels = labels
#         print("test:", self.test.shape)
#         print("train:", self.train.shape)

#     def __len__(self):
#         """
#         Number of images in the object dataset.
#         """
#         if self.flag == "train":
#             return (self.train.shape[0] - self.win_size) // self.step + 1
#         elif (self.flag == 'val'):
#             return (self.val.shape[0] - self.win_size) // self.step + 1
#         elif (self.flag == 'test'):
#             return (self.test.shape[0] - self.win_size) // self.step + 1
#         else:
#             return (self.test.shape[0] - self.win_size) // self.win_size + 1

#     def __getitem__(self, index):
#         index = index * self.step
#         if self.flag == "train":
#             return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
#         elif (self.flag == 'val'):
#             return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
#         elif (self.flag == 'test'):
#             return np.float32(self.test[index:index + self.win_size]), np.float32(
#                 self.test_labels[index:index + self.win_size])
#         else:
#             return np.float32(self.test[
#                               index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
#                 self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


# class UEAloader(Dataset):
#     """
#     Dataset class for datasets included in:
#         Time Series Classification Archive (www.timeseriesclassification.com)
#     Argument:
#         limit_size: float in (0, 1) for debug
#     Attributes:
#         all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
#             Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
#         feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
#         feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
#         all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
#         labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
#         max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
#             (Moreover, script argument overrides this attribute)
#     """

#     def __init__(self, root_path, file_list=None, limit_size=None, flag=None):
#         self.root_path = root_path
#         self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)
#         self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

#         if limit_size is not None:
#             if limit_size > 1:
#                 limit_size = int(limit_size)
#             else:  # interpret as proportion if in (0, 1]
#                 limit_size = int(limit_size * len(self.all_IDs))
#             self.all_IDs = self.all_IDs[:limit_size]
#             self.all_df = self.all_df.loc[self.all_IDs]

#         # use all features
#         self.feature_names = self.all_df.columns
#         self.feature_df = self.all_df

#         # pre_process
#         normalizer = Normalizer()
#         self.feature_df = normalizer.normalize(self.feature_df)
#         print(len(self.all_IDs))

#     def load_all(self, root_path, file_list=None, flag=None):
#         """
#         Loads datasets from csv files contained in `root_path` into a dataframe, optionally choosing from `pattern`
#         Args:
#             root_path: directory containing all individual .csv files
#             file_list: optionally, provide a list of file paths within `root_path` to consider.
#                 Otherwise, entire `root_path` contents will be used.
#         Returns:
#             all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
#             labels_df: dataframe containing label(s) for each sample
#         """
#         # Select paths for training and evaluation
#         if file_list is None:
#             data_paths = glob.glob(os.path.join(root_path, '*'))  # list of all paths
#         else:
#             data_paths = [os.path.join(root_path, p) for p in file_list]
#         if len(data_paths) == 0:
#             raise Exception('No files found using: {}'.format(os.path.join(root_path, '*')))
#         if flag is not None:
#             data_paths = list(filter(lambda x: re.search(flag, x), data_paths))
#         input_paths = [p for p in data_paths if os.path.isfile(p) and p.endswith('.ts')]
#         if len(input_paths) == 0:
#             raise Exception("No .ts files found using pattern: '{}'".format(pattern))

#         all_df, labels_df = self.load_single(input_paths[0])  # a single file contains dataset

#         return all_df, labels_df

#     def load_single(self, filepath):
#         df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
#                                                              replace_missing_vals_with='NaN')
#         labels = pd.Series(labels, dtype="category")
#         self.class_names = labels.cat.categories
#         labels_df = pd.DataFrame(labels.cat.codes,
#                                  dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

#         lengths = df.applymap(
#             lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series

#         horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

#         if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
#             df = df.applymap(subsample)

#         lengths = df.applymap(lambda x: len(x)).values
#         vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
#         if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
#             self.max_seq_len = int(np.max(lengths[:, 0]))
#         else:
#             self.max_seq_len = lengths[0, 0]

#         # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
#         # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
#         # sample index (i.e. the same scheme as all datasets in this project)

#         df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
#             pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

#         # Replace NaN values
#         grp = df.groupby(by=df.index)
#         df = grp.transform(interpolate_missing)

#         return df, labels_df

#     def instance_norm(self, case):
#         if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
#             mean = case.mean(0, keepdim=True)
#             case = case - mean
#             stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
#             case /= stdev
#             return case
#         else:
#             return case

#     def __getitem__(self, ind):
#         return self.instance_norm(torch.from_numpy(self.feature_df.loc[self.all_IDs[ind]].values)), \
#                torch.from_numpy(self.labels_df.loc[self.all_IDs[ind]].values)

#     def __len__(self):
#         return len(self.all_IDs)
