import numpy as np

def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true)) * 100

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true)) * 100

def metric(pred, true, scaler=None):
    """添加scaler参数进行逆变换"""
    if scaler is not None:
        pred = scaler.inverse_transform(pred)  # 逆变换到原始量纲
        true = scaler.inverse_transform(true)
    
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    
    # 处理MAPE/MSPE的除零问题
    mask = true != 0  # 忽略真实值为0的情况
    mape = np.mean(np.abs((pred[mask] - true[mask]) / true[mask])) * 100 if any(mask) else 0
    mspe = np.mean(np.square((pred[mask] - true[mask]) / true[mask])) * 100 if any(mask) else 0
    
    return mae, mse, rmse, mape, mspe