import time
from functools import partial

import dill
import numpy as np
import pandas as pd
import shap2
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split


def predict_function(x: np.ndarray, model: nn.Module, x_scaler, y_scaler, batch_size=128) -> np.ndarray:
    parameter = next(model.parameters())
    dtype = parameter.dtype
    device = parameter.device

    y_pred = list()
    while len(x) > 0:
        x_batch = x[:batch_size, ...]
        x = x[batch_size:, ...]

        scaled_x = torch.from_numpy(x_scaler.transform(x_batch)).to(dtype=dtype, device=device)
        scaled_y_pred = model(scaled_x)
        scaled_y_pred = scaled_y_pred.detach().cpu().numpy()
        y_pred.append(y_scaler.inverse_transform(scaled_y_pred))
    y_pred = np.vstack(y_pred)

    return y_pred


def main():
    rawdata = pd.read_csv("../data/housing.csv", delim_whitespace=True, header=None).values
    data = rawdata[..., :-1]
    target = rawdata[..., -1:].reshape(-1, 1)

    x_train, x_val, y_train, y_val = train_test_split(data, target, test_size=0.1, random_state=42)

    with open("../model/x_scaler.pkl", "rb") as f:
        x_scaler = dill.load(f)
    with open("../model/y_scaler.pkl", "rb") as f:
        y_scaler = dill.load(f)

    model = torch.load("../model/model.pt")
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    fn = partial(
        predict_function,
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        batch_size=256
    )

    explainer = shap2.KernelExplainer(
        model=fn,
        data=x_train
    )
    t0 = time.time()
    shap_values = explainer.shap_values(x_train)[0]
    t1 = time.time()

    np.save(f"kernel_shap_{(t1 - t0):.3f}.npy", shap_values)


if __name__ == "__main__":
    main()
