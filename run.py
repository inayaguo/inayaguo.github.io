import argparse
# import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
# from exp.exp_imputation import Exp_Imputation
# from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
# from exp.exp_anomaly_detection import Exp_Anomaly_Detection
# from exp.exp_classification import Exp_Classification
import random
import numpy as np
import pandas as pd
torch.manual_seed(seed=1)
torch.cuda.manual_seed_all(seed=1)
print(torch.cuda.is_available())
print(torch.cuda.device_count())

def start(feature_select, dimension_model, encoder_layers, batch_size, learning_rate, data_input, model, seed, i):
    fix_seed = seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimesNet')
    parser.add_argument('--set_seed', type=int, required=False, default=seed)

    # basic config
    parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')
    parser.add_argument('--model', type=str, required=False, default=model,
                        help='model name, options: [Autoformer, Transformer, TimesNet, FEDformer]')
    parser.add_argument('--month_predict', type=int, required=False, default=i)

    # data loader
    parser.add_argument('--data', type=str, required=False, default='sale', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default=data_input, help='data file')
    
    # time,slide,trends随意组合
    parser.add_argument('--features', type=str, default=feature_select,
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='predict', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='m',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=12, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=12, help='start token length')
    parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    
    # time:1,slide:7,trends:13
    feature_dimension = 1
    if 'slide' in feature_select:
        feature_dimension += 7
    if 'trends' in feature_select:
        feature_dimension += 13
    parser.add_argument('--enc_in', type=int, default=feature_dimension, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=feature_dimension, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=feature_dimension, help='output size')
    parser.add_argument('--time_in', type=int, default=1, help='output size')
    parser.add_argument('--slide_in', type=int, default=7, help='output size')
    parser.add_argument('--trends_in', type=int, default=13, help='output size')
    parser.add_argument('--d_time', type=int, default=64, help='output size')
    parser.add_argument('--d_slide', type=int, default=64, help='output size')
    parser.add_argument('--d_trends', type=int, default=64, help='output size')

    parser.add_argument('--d_model', type=int, default=dimension_model, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=encoder_layers, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=11, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=batch_size, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=learning_rate, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args.gpu)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    # elif args.task_name == 'short_term_forecast':
    #     Exp = Exp_Short_Term_Forecast
    # elif args.task_name == 'imputation':
    #     Exp = Exp_Imputation
    # elif args.task_name == 'anomaly_detection':
    #     Exp = Exp_Anomaly_Detection
    # elif args.task_name == 'classification':
    #     Exp = Exp_Classification
    else:
        Exp = Exp_Long_Term_Forecast

    
    if args.is_training:
        for ii in range(args.itr):
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()

if __name__ == '__main__':
    # 实验
    # feature_select = 'time_trends_slide'
    # for data_input in ['data_mj.csv']: # todo1:'data_m5.csv', , 'data_kimberly.csv'
    #     if 'mj' in data_input:
    #         model_list = ['TimesNet'] # 'TimesNet', 'MLP', 'DLinear', 'LightTS', 'Autoformer', 'MLP', 'DLinear', 'LightTS', 'Autoformer', 'FEDformer', 'Saleformer', 
    #     else:
    #         model_list = ['MLP', 'DLinear', 'LightTS', 'Autoformer', 'FEDformer', 'Saleformer', 'TimesNet'] # 'LSTM', 'MLP', 'DLinear', 'LightTS', 'Autoformer', 
    #     for model in model_list:
    #         if (model == 'MLP') and ('mj' in data_input):
    #             list_seed = [2021, 42, 2035, 1908, 1809, 2024, 2012, 2001, 1974, 10, 882, 11152, 11670, 15171, 18702, 20642, 21880, 22148, 28672, 28753]
    #         else:
    #             list_seed = [2021, 42, 2035, 1908, 1809, 2024, 2012, 2001, 1974, 10, 882, 11152, 11670, 15171, 18702, 20642, 21880, 22148, 28672, 28753]
    #         for seed in list_seed:
    #             for learning_rate in [5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]:
    #                 batch_size = 64
    #                 encoder_layers = 2
    #                 decoder_layers = 1
    #                 if model == 'TimesNet':
    #                     dimension_model = 16
    #                 else:
    #                     dimension_model = 512
    #                 if 'mj' in data_input: # 美赞
    #                     for month in range(202108, 202113): # 5个月
    #                         start(feature_select, dimension_model, encoder_layers, batch_size, learning_rate, data_input, model, seed, month)
    #                     for month in range(202201, 202204): # 3个月
    #                         start(feature_select, dimension_model, encoder_layers, batch_size, learning_rate, data_input, model, seed, month)
    #                 elif 'kimberly' in data_input: # 金佰利
    #                     for month in range(202208, 202213): # 5个月
    #                         start(feature_select, dimension_model, encoder_layers, batch_size, learning_rate, data_input, model, seed, month)
    #                     else: # m5
    #                         for month in range(202208, 202213):
    #                             start(feature_select, dimension_model, encoder_layers, batch_size, learning_rate, data_input, model, seed, month)
    '''清洗训练集&测试集'''
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


    input_data = 'deep_train_202305.csv'
    start(feature_select='time_trends_slide', dimension_model=128, encoder_layers=2, batch_size=16,
          learning_rate=0.005, data_input=input_data, model='FEDformer', seed=2021, i=202305)

    '''测试参数'''
    '''
    def test_params():
        for dimension_model in [256, 512]:
            for batch_size in [32, 64, 128]:
                for learning_rate in [1e-3, 1e-4, 5e-4]:
                    # 美赞验收
                    for month_data in range(202208, 202213):    
                        input_data = 'deep_train_mz_' + str(month_data) + '.csv'
                        start(feature_select = 'time_trends_slide', dimension_model=dimension_model, encoder_layers=2, batch_size=batch_size, learning_rate=learning_rate, data_input=input_data, model='FEDformer', seed=2021, i=month_data)
                    
                    for month_data in range(202301, 202313):    
                        input_data = 'deep_train_mz_' + str(month_data) + '.csv'
                        start(feature_select = 'time_trends_slide', dimension_model=dimension_model, encoder_layers=2, batch_size=batch_size, learning_rate=learning_rate, data_input=input_data, model='FEDformer', seed=2021, i=month_data)

                    for month_data in range(202401, 202404):
                        input_data = 'deep_train_mz_' + str(month_data) + '.csv'
                        start(feature_select = 'time_trends_slide', dimension_model=dimension_model, encoder_layers=2, batch_size=batch_size, learning_rate=learning_rate, data_input=input_data, model='FEDformer', seed=2021, i=month_data)
    '''
    # test_params()

    '''选择最优参数训练并输出结果'''
    '''
    parameters = pd.read_csv('/home/ubuntu/Time-Series-Library-main/parameters_selected_0.3.csv', encoding='utf_8_sig')
    for month_data in range(202208, 202213):
        filtered_parameters = parameters[parameters['month'] == month_data]
        input_data = 'deep_train_mz_' + str(month_data) + '.csv'
        start(feature_select = 'time_trends_slide', dimension_model=filtered_parameters['dimension_model'].item(), encoder_layers=2, batch_size=filtered_parameters['batch_size'].item(), learning_rate=filtered_parameters['learning_rate'].item(), data_input=input_data, model='FEDformer', seed=2021, i=month_data)
                
    for month_data in range(202301, 202313):    
        filtered_parameters = parameters[parameters['month'] == month_data]
        input_data = 'deep_train_mz_' + str(month_data) + '.csv'
        start(feature_select = 'time_trends_slide', dimension_model=filtered_parameters['dimension_model'].item(), encoder_layers=2, batch_size=filtered_parameters['batch_size'].item(), learning_rate=filtered_parameters['learning_rate'].item(), data_input=input_data, model='FEDformer', seed=2021, i=month_data)

    for month_data in range(202401, 202404):
        filtered_parameters = parameters[parameters['month'] == month_data]
        input_data = 'deep_train_mz_' + str(month_data) + '.csv'
        start(feature_select = 'time_trends_slide', dimension_model=filtered_parameters['dimension_model'].item(), encoder_layers=2, batch_size=filtered_parameters['batch_size'].item(), learning_rate=filtered_parameters['learning_rate'].item(), data_input=input_data, model='FEDformer', seed=2021, i=month_data)
    '''

    # for month in range(202208, 202213):
    #     start(feature_select = 'time_trends_slide', dimension_model=256, encoder_layers=2, batch_size=64, learning_rate=1e-4, data_input='deep_train_mz_' + str(month) + '.csv', model='FEDformer', seed=2021, i=month)
    # for month in range(202301, 202313):
    #     start(feature_select = 'time_trends_slide', dimension_model=256, encoder_layers=2, batch_size=64, learning_rate=1e-4, data_input='deep_train_mz_' + str(month) + '.csv', model='FEDformer', seed=2021, i=month)
    # for month in range(202401, 202404):
        # start(feature_select = 'time_trends_slide', dimension_model=256, encoder_layers=2, batch_size=64, learning_rate=1e-4, data_input='deep_train_mz_' + str(month) + '.csv', model='FEDformer', seed=2021, i=month)


# for batch_size in [128, 256]:
#                         for encoder_layers in range(2, 4):
#                             for decoder_layers in range(1,3):
#                                 for dimension_model in [128, 256, 512]: