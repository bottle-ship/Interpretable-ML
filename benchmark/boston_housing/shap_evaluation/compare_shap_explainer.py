import dill
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def main():
    rawdata = pd.read_csv("../data/housing.csv", delim_whitespace=True, header=None).values
    data = rawdata[..., :-1]
    target = rawdata[..., -1:].reshape(-1, 1)

    x_train, x_val, y_train, y_val = train_test_split(data, target, test_size=0.1, random_state=42)

    with open("../model/x_scaler.pkl", "rb") as f:
        x_scaler = dill.load(f)
    with open("../model/y_scaler.pkl", "rb") as f:
        y_scaler = dill.load(f)

    shap_values_kernel = np.load("./kernel_shap_1271.424.npy")
    shap_values_scaled_kernel = np.load("./scaled_kernel_shap_1028.316.npy")
    shap_values_deep = np.load("./deep_shap_4.348.npy")
    shap_values_gradient = np.load("./gradient_shap_15.474.npy")
    shap_values_captum_kernel = np.load("./captum_kernel_shap_510.855.npy")

    print(np.sum(np.abs(shap_values_kernel)))  # 9306.001220398395
    print(np.sum(np.abs(shap_values_scaled_kernel * y_scaler.scale_)))  # 9314.254760567786
    print(np.sum(np.abs(shap_values_deep * y_scaler.scale_)))  # 7798.9176777083485
    print(np.sum(np.abs(shap_values_gradient * y_scaler.scale_)))  # 7755.550475653344
    print(np.sum(np.abs(shap_values_captum_kernel)))  # 8084.5645

    x = np.arange(0, data.shape[1])
    width = 0.15

    plt.figure()
    plt.bar(x - width * 1.5, np.mean(np.abs(shap_values_kernel), axis=0), width=width,
            label="Kernel")
    plt.bar(x - width / 2, np.mean(np.abs(shap_values_scaled_kernel * y_scaler.scale_), axis=0), width=width,
            label="Scaled Kernel")
    plt.bar(x + width / 2, np.mean(np.abs(shap_values_deep * y_scaler.scale_), axis=0), width=width,
            label="Deep")
    plt.bar(x + width * 1.5, np.mean(np.abs(shap_values_gradient * y_scaler.scale_), axis=0), width=width,
            label="Gradient")
    plt.bar(x + width * 3.0, np.mean(np.abs(shap_values_captum_kernel), axis=0), width=width,
            label="Kernel (Captum)")

    plt.legend()
    plt.savefig("compare_shap_explainer.png")


if __name__ == "__main__":
    main()
