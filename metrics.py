import torch


def calculate_point_fidelity(
        base_model_pred,
        surrogate_model_pred,
):
    return torch.pow(surrogate_model_pred - base_model_pred, 2)


# x's shape - [batch_size, features]
def mtl_loss(
        x,
        y,
        base_loss_fn,
        alpha,
        base_model_pred,
        surrogate_model_pred
):
    n_elements = x.shape[0]
    base_loss_term = alpha * base_loss_fn(base_model_pred, y)
    point_fidelity_term = (1 - alpha) * calculate_point_fidelity(base_model_pred, surrogate_model_pred)
    return (1 / n_elements) * torch.sum(base_loss_term + point_fidelity_term, dim=-1)