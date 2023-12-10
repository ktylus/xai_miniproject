import torch


def calculate_point_fidelity(
        base_model_pred,
        surrogate_model_pred,
):
    return torch.pow(surrogate_model_pred - base_model_pred, 2)


def calculate_global_fidelity(
        base_model_pred,
        surrogate_model_pred
):
    n_elements = base_model_pred.shape[0]
    return (1 / n_elements) * torch.sum(calculate_point_fidelity(base_model_pred, surrogate_model_pred), dim=-1)


def calculate_neighborhood_fidelity(
        base_model,
        surrogate_model,
        x,
        perturbation_variance,
        n_points
):
    # x.shape is: [batch_size, n_features]
    variance = perturbation_variance * torch.eye(x.shape[1])
    nearby_points = torch.normal(x, variance, n_points)
    return (1 / n_points) * torch.sum(torch.pow(surrogate_model(nearby_points) - base_model(nearby_points), 2), dim=-1)


def calculate_global_neighborhood_fidelity(
        base_model,
        surrogate_model,
        x,
        perturbation_variance,
        n_points
):
    return torch.sum(calculate_neighborhood_fidelity(base_model, surrogate_model, x,
                                                     perturbation_variance, n_points), dim=-1)


def mtl_loss(
        x,
        y,
        base_loss_fn,
        alpha,
        base_model_pred,
        surrogate_model_pred
):
    # x.shape is: [batch_size, n_features]
    n_elements = x.shape[0]
    base_loss_term = alpha * base_loss_fn(base_model_pred, y)
    point_fidelity_term = (1 - alpha) * calculate_point_fidelity(base_model_pred, surrogate_model_pred)
    return (1 / n_elements) * torch.sum(base_loss_term + point_fidelity_term, dim=-1)
