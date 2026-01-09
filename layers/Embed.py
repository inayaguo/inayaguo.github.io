import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
import csv
import numpy as np

class SlideEmbedding(nn.Module):
    def __init__(self, slide_in, d_slide):
        super(SlideEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=slide_in, out_channels=d_slide,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class TrendsEmbedding(nn.Module):
    def __init__(self, trends_in, d_trends):
        super(TrendsEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=trends_in, out_channels=d_trends,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x
    
class TimeEmbedding(nn.Module):
    def __init__(self, time_in, d_time):
        super(TimeEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=time_in, out_channels=d_time,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x
    
class Token2DEmbedding(nn.Module):
    def __init__(self, c_in, c_out):
        super(Token2DEmbedding, self).__init__()
        self.tokenConv = nn.Conv2d(in_channels=c_in, out_channels=c_out,
                                   kernel_size=(3,2), bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x)
        return x

class SaleEmbedding(nn.Module):
    def __init__(self, time_in, trends_in, slide_in, d_slide, d_time, d_trends, d_model, c_in, d2_in_out, features_input, embed_type='fixed', freq='h', dropout=0.1):
        super(SaleEmbedding, self).__init__()

        # 1.1 原始多维特征卷积embedding（ConV1d）
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        # 1.2 位置embedding
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        # 1.3 时序embedding
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        '''新增embedding'''
        # 2:对特征进行一维卷积
        # 2.1 时序数据embedding（ConV1d）
        self.time_embedding = TimeEmbedding(time_in=time_in, d_time=d_time)
        # 2.2 滑动窗口embedding（ConV1d）
        self.slide_embedding = SlideEmbedding(slide_in=slide_in, d_slide=d_slide)
        # 2.3 销量趋势embedding（ConV1d）
        self.trends_embedding = TrendsEmbedding(trends_in=trends_in, d_trends=d_trends)
        # 3:对三类特征进行二维卷积
        # 3.1 三类特征embedding（ConV2d）
        # todo:kernel_size需要随输入发生变化
        self.features_embedding = Token2DEmbedding(c_in=d2_in_out, c_out=d2_in_out)
        # 3.2 处理为模型所需维度（ConV1d）
        self.features_value_embedding = TokenEmbedding(c_in=63, d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.features_input = features_input

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x) + torch.concat(self.time_embedding(x), self.slide_embedding(x), self.trends_embedding(x))
        else:
            # self.time_embedding/self.slide_embedding/self.trends_embedding输出tensor形状为[batch_size, seq_len, d_time/d_slide/d_trends]
            input_data = self.time_embedding(x[:,:,0:1]).unsqueeze(2)
            if 'trends' in self.features_input:
                input_data = torch.concat((input_data, self.trends_embedding(x[:,:,8:]).unsqueeze(2)), dim=2)
            if 'slide' in self.features_input:
                input_data = torch.concat((input_data, self.slide_embedding(x[:,:,1:8]).unsqueeze(2)), dim=2)
            # features_embed形状为[batch_size, seq_len, 3(特征类别数量), d_time/d_slide/d_trends]
            features_embed = torch.squeeze(self.features_embedding(input_data), dim=2)
            x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x) + self.features_value_embedding(features_embed)
            # x形状为[batch_size, seq_len, d_model]
        return self.dropout(x)

class change_process(nn.Module):
    def __init__(self, time_in, trends_in, slide_in, d_slide, d_time, d_trends, d_model, c_in, d2_in_out, features_input, embed_type='fixed', freq='h', dropout=0.1):
        super(change_process, self).__init__()

        '''新增embedding'''
        # 2:对特征进行一维卷积
        # 2.1 时序数据embedding（ConV1d）
        self.time_embedding = TimeEmbedding(time_in=time_in, d_time=d_time)
        # 2.2 滑动窗口embedding（ConV1d）
        self.slide_embedding = SlideEmbedding(slide_in=slide_in, d_slide=d_slide)
        # 2.3 销量趋势embedding（ConV1d）
        self.trends_embedding = TrendsEmbedding(trends_in=trends_in, d_trends=d_trends)
        # 3:对三类特征进行二维卷积
        # 3.1 三类特征embedding（ConV2d）
        # todo:kernel_size需要随输入发生变化
        self.features_embedding = Token2DEmbedding(c_in=d2_in_out, c_out=d2_in_out)
        # 3.2 处理为模型所需维度（ConV1d）
        self.features_value_embedding = TokenEmbedding(c_in=63, d_model=21)
        self.dropout = nn.Dropout(p=dropout)
        self.features_input = features_input

    def forward(self, x):
        # self.time_embedding/self.slide_embedding/self.trends_embedding输出tensor形状为[batch_size, seq_len, d_time/d_slide/d_trends]
        input_data = self.time_embedding(x[:,:,0:1]).unsqueeze(2)
        if 'trends' in self.features_input:
            input_data = torch.concat((input_data, self.trends_embedding(x[:,:,8:]).unsqueeze(2)), dim=2)
        if 'slide' in self.features_input:
            input_data = torch.concat((input_data, self.slide_embedding(x[:,:,1:8]).unsqueeze(2)), dim=2)
        # features_embed形状为[batch_size, seq_len, 3(特征类别数量), d_time/d_slide/d_trends]
        features_embed = torch.squeeze(self.features_embedding(input_data), dim=2)
        x = self.features_value_embedding(features_embed)
        # x形状为[batch_size, seq_len, d_model]
        return self.dropout(x)

# 深度可分离卷积（CVPR2017）
class DeepWise_PointWise_Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DeepWise_PointWise_Conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)

    # 新增：保存DataEmbedding所有参数到CSV
    def save_embedding_params(self, save_path="dataembedding_params.csv"):
        """
        保存DataEmbedding层的所有可训练参数到CSV文件
        :param save_path: CSV文件保存路径
        """
        # 获取所有参数（value_embedding/position_embedding/temporal_embedding）
        params = self.named_parameters()
        csv_header = ["param_name", "param_shape", "param_values"]
        csv_data = []

        for name, param in params:
            # 转换为numpy数组并展平（便于CSV存储）
            param_np = param.cpu().detach().numpy()
            param_flat = param_np.flatten()
            # 格式化为字符串（保留6位小数）
            param_str = ",".join([f"{v:.6f}" for v in param_flat])
            csv_data.append([name, str(param_np.shape), param_str])

        # 写入CSV
        with open(save_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)
            writer.writerows(csv_data)

        print(f"✅ DataEmbedding参数已保存至: {save_path}")

    # 新增：提取核心权重（用于特征贡献分析）
    def get_core_weights(self):
        """
        获取value_embedding的核心卷积权重（TokenEmbedding的Conv1d权重）
        返回：dict，包含权重矩阵和特征贡献度
        """
        # 获取TokenEmbedding的Conv1d权重
        conv_weight = self.value_embedding.tokenConv.weight.cpu().detach().numpy()  # 形状: [d_model, c_in, kernel_size]
        # 计算每个输入特征的权重绝对值均值（量化贡献度）
        # conv_weight形状：[out_channels(d_model), in_channels(c_in), kernel_size]
        feature_contribution = np.mean(np.abs(conv_weight), axis=(0, 2))  # 按输入特征维度求均值，形状: [c_in,]

        return {
            "conv_weight": conv_weight,  # 卷积核权重
            "feature_contribution": feature_contribution,  # 各输入特征贡献度
            "position_embedding": self.position_embedding.pe.cpu().detach().numpy()  # 位置编码参数
        }


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars
