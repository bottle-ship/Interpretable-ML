import dill
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from iml.datasets import XYDataset
from iml.model import FullyConnectedNeuralNetwork


def compute_loss(x: torch.Tensor, y: torch.Tensor, model: nn.Module, criterion: nn.Module) -> np.ndarray:
    y_hat = model(x)
    loss = criterion(y_hat, y).item()

    return loss


def main():
    max_epochs = 200

    rawdata = pd.read_csv("../data/housing.csv", delim_whitespace=True, header=None).values
    data = rawdata[..., :-1]
    target = rawdata[..., -1:].reshape(-1, 1)

    x_train, x_val, y_train, y_val = train_test_split(data, target, test_size=0.1, random_state=42)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    scaled_x_train = x_scaler.fit_transform(x_train)
    scaled_y_train = y_scaler.fit_transform(y_train)
    scaled_x_val = x_scaler.transform(x_val)
    scaled_y_val = y_scaler.transform(y_val)

    with open("./x_scaler.pkl", "wb") as f:
        dill.dump(x_scaler, f)
    with open("./y_scaler.pkl", "wb") as f:
        dill.dump(y_scaler, f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaled_x_train_torch = torch.from_numpy(scaled_x_train).to(dtype=torch.float32, device=device)
    scaled_y_train_torch = torch.from_numpy(scaled_y_train).to(dtype=torch.float32, device=device)
    scaled_x_val_torch = torch.from_numpy(scaled_x_val).to(dtype=torch.float32, device=device)
    scaled_y_val_torch = torch.from_numpy(scaled_y_val).to(dtype=torch.float32, device=device)

    model = FullyConnectedNeuralNetwork(
        n_inputs=x_train.shape[1],
        n_outputs=y_train.shape[1],
        activation="SiLU"
    ).float()
    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_dataset = XYDataset(x=scaled_x_train_torch, y=scaled_y_train_torch)
    train_dataloaders = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

    for epoch in range(0, max_epochs):
        model.train()
        for i, data in enumerate(train_dataloaders):
            x_batch, y_batch = data

            optimizer.zero_grad()

            y_batch_hat = model.forward(x_batch)
            loss_batch = criterion(y_batch_hat, y_batch)
            loss_batch.backward()
            optimizer.step()
        model.eval()

        loss_train = compute_loss(scaled_x_train_torch, scaled_y_train_torch, model, criterion)
        loss_val = compute_loss(scaled_x_val_torch, scaled_y_val_torch, model, criterion)

        print(loss_train, loss_val, flush=True)

    y_train_pred = y_scaler.inverse_transform(model.forward(scaled_x_train_torch).detach().cpu().numpy())
    y_val_pred = y_scaler.inverse_transform(model.forward(scaled_x_val_torch).detach().cpu().numpy())

    import matplotlib.pyplot as plt

    plt.figure()
    plt.scatter(y_train.ravel(), y_train_pred.ravel())
    plt.scatter(y_val.ravel(), y_val_pred.ravel())
    plt.savefig("scatter.png")

    model.cpu()
    torch.save(model, "model.pt")


if __name__ == "__main__":
    main()
