from torch import nn
import torch.nn.functional as F


class BaseClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, n_hidden_layers, layer_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_layer = nn.Linear(input_dim, layer_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(layer_size, layer_size) for _ in range(n_hidden_layers - 1)])
        self.output_layer = nn.Linear(layer_size, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = F.relu(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x


class BaseRegressor(nn.Module):
    def __init__(self, input_dim, output_dim, n_hidden_layers, layer_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_layer = nn.Linear(input_dim, layer_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(layer_size, layer_size) for _ in range(n_hidden_layers - 1)])
        self.output_layer = nn.Linear(layer_size, output_dim)

    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = F.relu(x)
        x = self.output_layer(x)
        return x
