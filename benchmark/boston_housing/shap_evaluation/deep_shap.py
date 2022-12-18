import time

import dill
import numpy as np
import pandas as pd
import shap2
import torch
from sklearn.model_selection import train_test_split


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

    parameter = next(model.parameters())
    dtype = parameter.dtype
    device = parameter.device

    scaled_x_train = torch.from_numpy(x_scaler.transform(x_train)).to(dtype=dtype, device=device)

    deep_pytorch = getattr(getattr(shap2.explainers, "_deep"), "deep_pytorch")
    op_handler = getattr(deep_pytorch, "op_handler")
    op_handler["SiLU"] = getattr(deep_pytorch, "nonlinear_1d")

    explainer = shap2.DeepExplainer(
        model=model,
        data=scaled_x_train
    )
    t0 = time.time()
    shap_values = explainer.shap_values(scaled_x_train)
    t1 = time.time()

    np.save(f"deep_shap_{(t1 - t0):.3f}.npy", shap_values)


if __name__ == "__main__":
    main()
