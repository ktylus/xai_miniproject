from torch import nn


class SurrogateClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear_layer = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear_layer(x)
        x = self.sigmoid(x)
        return x


class SurrogateRegressor(nn.Module):
    def __init__(self, input_dim, output_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear_layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear_layer(x)
        return x
