import numpy as np


def life_metrics(pred_life, true_life):
    err = pred_life - true_life
    return {"rmse": float(np.sqrt((err ** 2).mean())), "mape": float(np.abs(err / true_life).mean() * 100),
            "r2": float(1 - (err ** 2).sum() / ((true_life - true_life.mean()) ** 2).sum())}
