from torch import nn


class SurrogateModel(nn.Module):
    def __init__(self, input_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.linear_layer = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear_layer(x)
