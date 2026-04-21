import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def nse(target, pred):
    return 1 - (np.sum((target - pred) ** 2) / np.sum((target - np.mean(target)) ** 2))


def mae(target, pred):
    return mean_absolute_error(target, pred)


def rmse(target, pred):
    return root_mean_squared_error(target, pred)



def smape(target, pred):
    target = np.array(target)
    pred = np.array(pred)
    
    denom = (np.abs(target) + np.abs(pred)) / 2
    
    # Check for division by zero
    if np.any(denom == 0):
        return None
    
    return 100 / len(target) * np.sum(np.abs(target - pred) / denom)


def mean_smape_over_horizons(y_target, y_pred):
    smapes = []
    for i in range(y_target.shape[1]):
        smapes.append(smape(y_target[:, i], y_pred[:, i]))
    return np.mean(smapes)


def evaluate_forecast(target, pred):
    results = {}
    results["nse"] = nse(target, pred)
    results["mae"] = mae(target, pred)
    results["rmse"] = root_mean_squared_error(target, pred)
    results["smape"] = smape(target, pred)
    return results