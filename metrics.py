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
        perturbation_variance,
        n_points
):
    variance = perturbation_variance * torch.eye(x.shape[1])
    nearby_points = torch.normal(x, variance, n_points)
    base_model_preds = base_model(nearby_points)
    surrogate_model_preds = surrogate_model(nearby_points)
    return calculate_global_fidelity(base_model_preds, surrogate_model_preds)


def calculate_global_neighborhood_fidelity(
        base_model,
        surrogate_model,
        x,
        perturbation_variance,
        n_points
):
    return torch.sum(calculate_neighborhood_fidelity(base_model, surrogate_model, x,
                                                     perturbation_variance, n_points))
