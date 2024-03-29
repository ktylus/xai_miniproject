import torch
import math

# code from:
# https://datascience.stackexchange.com/questions/96271/logcoshloss-on-pytorch
# that is based on keras implementation:
# https://github.com/keras-team/keras/blob/v2.6.0/keras/losses.py#L1580-L1617

def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        return x + torch.nn.functional.softplus(-2.0 * x) - math.log(2.0)

    return torch.mean(_log_cosh(y_pred - y_true))

# logarithm of the hyperbolic cosine
class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return log_cosh_loss(y_pred, y_true)
