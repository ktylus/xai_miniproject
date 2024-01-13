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
    return (1 / n_elements) * torch.sum(calculate_point_fidelity(base_model_pred, surrogate_model_pred))


def calculate_neighborhood_fidelity(
        base_model,
        surrogate_model,
        x,
        perturbation_variance=0.1,
        n_points=50
):
    x_expanded = x.unsqueeze(-1).expand(*x.shape, n_points)
    nearby_points = torch.normal(x_expanded, perturbation_variance)
    nearby_points = torch.swapaxes(nearby_points, 1, 2)
    base_model_preds = base_model(nearby_points).squeeze()
    surrogate_model_preds = surrogate_model(nearby_points).squeeze()
    return (1 / n_points) * torch.sum(torch.pow(surrogate_model_preds - base_model_preds, 2), dim=-1)


def calculate_global_neighborhood_fidelity(
        base_model,
        surrogate_model,
        x,
        perturbation_variance=0.1,
        n_points=50
):
    batch_size = x.shape[0]
    return (1 / batch_size) * torch.sum(calculate_neighborhood_fidelity(base_model, surrogate_model, x,
                                                     perturbation_variance, n_points))
