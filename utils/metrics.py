import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    # 检查并移除 NaN 值和无限值
    valid_mask = np.isfinite(pred) & np.isfinite(true)
    pred = pred[valid_mask]
    true = true[valid_mask]

    # 确保输入数据不为空
    if len(pred) == 0 or len(true) == 0:
        return 0

    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    # 检查并移除 NaN 值和无限值
    valid_mask = np.isfinite(pred) & np.isfinite(true)
    pred = pred[valid_mask]
    true = true[valid_mask]

    # 确保输入数据不为空
    if len(pred) == 0 or len(true) == 0:
        return 0
        
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    non_zero_indices = true != 0
    y_true_filtered = true[non_zero_indices]
    y_pred_filtered = pred[non_zero_indices]
    return np.mean(np.abs((y_pred_filtered - y_true_filtered) / y_true_filtered))


def MSPE(pred, true):
    non_zero_indices = true != 0
    y_true_filtered = true[non_zero_indices]
    y_pred_filtered = pred[non_zero_indices]
    return np.mean(np.square((y_pred_filtered - y_true_filtered) / y_true_filtered))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe
