import typing as t

import torch
import torch.nn as nn

__all__ = ["FullyConnectedNeuralNetwork"]


class FullyConnectedNeuralNetwork(nn.Module):

    def __init__(
            self,
            n_inputs: int,
            n_outputs: int,
            hidden_layer_sizes: t.Sequence[int] = (256, 256, 256),
            activation: t.Union[str, nn.Module] = "ReLU"
    ):
        super(FullyConnectedNeuralNetwork, self).__init__()

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.hidden_layer_sizes = hidden_layer_sizes

        if isinstance(activation, nn.Module):
            self.activation = activation
        else:
            self.activation = getattr(nn, activation)()

        self.layers = torch.nn.ModuleList(
            nn.Linear(
                in_features=self.n_inputs if i == 0 else self.hidden_layer_sizes[i - 1],
                out_features=self.hidden_layer_sizes[i]
            )
            for i in range(0, len(self.hidden_layer_sizes))
        )
        self.output_layer = nn.Linear(
            in_features=self.hidden_layer_sizes[-1],
            out_features=self.n_outputs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        return self.output_layer(x)
