import time

import dill
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from captum import attr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class PredictFunction(nn.Module):

    def __init__(
            self,
            model: nn.Module,
            x_scaler: StandardScaler,
            y_scaler: StandardScaler
    ):
        super(PredictFunction, self).__init__()

        self.model = model
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler

        parameter = next(model.parameters())
        dtype = parameter.dtype
        device = parameter.device

        self._x_scaler_mean_ = torch.from_numpy(self.x_scaler.mean_).to(dtype=dtype, device=device)
        self._x_scaler_scale_ = torch.from_numpy(self.x_scaler.scale_).to(dtype=dtype, device=device)

        self._y_scaler_mean_ = torch.from_numpy(self.y_scaler.mean_).to(dtype=dtype, device=device)
        self._y_scaler_scale_ = torch.from_numpy(self.y_scaler.scale_).to(dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled_x = x - self._x_scaler_mean_
        scaled_x = scaled_x / self._x_scaler_scale_

        scaled_y_hat = self.model(scaled_x)

        y_hat = scaled_y_hat * self._y_scaler_scale_
        y_hat = y_hat + self._y_scaler_mean_

        return y_hat


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

    predict_function = PredictFunction(
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler
    )

    parameter = next(model.parameters())
    dtype = parameter.dtype
    device = parameter.device

    x_train_torch = torch.from_numpy(x_train).to(dtype=dtype, device=device)
    x_train_torch_mean = torch.mean(x_train_torch, dim=0)

    explainer = attr.KernelShap(predict_function)
    t0 = time.time()
    shap_values = explainer.attribute(
        inputs=x_train_torch,
        baselines=x_train_torch_mean,
        n_samples=2 * x_train_torch.shape[1] + 2048,
        show_progress=True
    ).detach().cpu().numpy()
    t1 = time.time()

    np.save(f"captum_kernel_shap_{(t1 - t0):.3f}.npy", shap_values)


if __name__ == "__main__":
    main()
