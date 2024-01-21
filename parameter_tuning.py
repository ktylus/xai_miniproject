from functools import partial

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import ray.tune as tune
from ray.air import session
from ray.tune.schedulers import ASHAScheduler
from log_cosh_loss import LogCoshLoss

from datasets import CaliforniaHousingDataset, AdultDataset, TitanicDataset, AutoMpgDataset, WineDataset
from metrics import calculate_global_fidelity
from models.base_model import BaseClassifier, BaseRegressor
from models.surrogate_model import SurrogateClassifier, SurrogateRegressor

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def load_housing_data():
    housing_train = CaliforniaHousingDataset(
        dataset_path="data/california_housing/cal_housing.data", normalize=True, train=True)
    housing_test = CaliforniaHousingDataset(
        dataset_path="data/california_housing/cal_housing.data", normalize=True, train=False)
    return housing_train, housing_test


def load_adult_data():
    adult_train = AdultDataset(dataset_path="data/adult/adult.data", normalize=True, train=True)
    adult_test = AdultDataset(dataset_path="data/adult/adult.data", normalize=True, train=False)
    return adult_train, adult_test


def load_wine_data():
    wine_train = WineDataset(dataset_path="data/wines/winequality-red.csv", normalize=True, train=True)
    wine_test = WineDataset(dataset_path="data/wines/winequality-red.csv", normalize=True, train=False)
    return wine_train, wine_test


def load_titanic_data():
    titanic_train = TitanicDataset(dataset_path="data/titanic/titanic.arff", normalize=True, train=True)
    titanic_test = TitanicDataset(dataset_path="data/titanic/titanic.arff", normalize=True, train=False)
    return titanic_train, titanic_test


def load_autompg_data():
    autompg_train = AutoMpgDataset(dataset_path="data/autompg/auto-mpg.data", normalize=True, train=True)
    autompg_test = AutoMpgDataset(dataset_path="data/autompg/auto-mpg.data", normalize=True, train=False)
    return autompg_train, autompg_test


def tuning_train(
        config: dict,
        train_data: Dataset,
        test_data: Dataset,
        criterion,
        epochs: int,
        alpha: float
):
    is_classification = criterion == binary_classification_criterion
    if is_classification:
        base_model = BaseClassifier(input_dim=train_data.features.shape[1],
            output_dim=1, n_hidden_layers=config["n_hidden_layers"], layer_size=config["layer_size"]).to(device)
        surrogate_model = SurrogateClassifier(input_dim=train_data.features.shape[1], output_dim=1).to(device)
    else:
        base_model = BaseRegressor(input_dim=train_data.features.shape[1],
            output_dim=1, n_hidden_layers=config["n_hidden_layers"], layer_size=config["layer_size"]).to(device)
        surrogate_model = SurrogateRegressor(input_dim=train_data.features.shape[1], output_dim=1).to(device)

    params = list(base_model.parameters()) + list(surrogate_model.parameters())
    optimizer = Adam(params, lr=lr)
    loader = DataLoader(train_data, batch_size=config["batch_size"])
    for epoch in range(epochs):
        running_loss = 0
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            labels = labels.reshape(-1, 1)
            optimizer.zero_grad()

            base_model_preds = base_model(data)
            surrogate_model_preds = surrogate_model(data)
            loss = criterion(base_model_preds, labels)
            point_fidelity = calculate_global_fidelity(base_model_preds, surrogate_model_preds)
            mtl_loss = alpha * loss + (1 - alpha) * point_fidelity

            mtl_loss.backward()
            optimizer.step()
            running_loss += mtl_loss

        valid_mtl_loss = calculate_validation_mtl_loss(
            base_model, surrogate_model, test_data, criterion, alpha).item()
        session.report({"loss": valid_mtl_loss})


def calculate_validation_mtl_loss(
        base_model: nn.Module,
        surrogate_model: nn.Module,
        test_data: Dataset,
        criterion,
        alpha: float
):
    test_loader = DataLoader(test_data, batch_size=len(test_data))
    data, labels = next(iter(test_loader))
    data, labels = data.to(device), labels.to(device)
    labels = labels.reshape((-1, 1))
    with torch.no_grad():
        base_model_preds = base_model(data)
        surrogate_model_preds = surrogate_model(data)
        loss = criterion(base_model_preds, labels)
        point_fidelity = calculate_global_fidelity(base_model_preds, surrogate_model_preds)
        mtl_loss = alpha * loss + (1 - alpha) * point_fidelity
        return mtl_loss


adult_train, adult_test = load_adult_data()
titanic_train, titanic_test = load_titanic_data()

classification_data = {
    "adult": (adult_train, adult_test),
    "titanic": (titanic_train, titanic_test)
}

wine_train, wine_test = load_wine_data()
housing_train, housing_test = load_housing_data()
autompg_train, autompg_test = load_autompg_data()

regression_data = {
    "wine": (wine_train, wine_test),
    "housing": (housing_train, housing_test),
    "autompg": (autompg_train, autompg_test)
}

lr = 0.001
binary_classification_criterion = torch.nn.BCELoss()
regression_criterion = LogCoshLoss() # "logarithm of the hyperbolic cosine" from the paper

epochs = 50
alpha = 0.5

config = {
    "n_hidden_layers": tune.choice([i for i in range(2, 7)]),
    "layer_size": tune.choice([2 ** i for i in range(4, 10)]),
    "batch_size": tune.choice([32, 64, 128])
}

scheduler = ASHAScheduler(
    metric="loss",
    mode="min",
    max_t=epochs,
    grace_period=1,
    reduction_factor=2,
)

tuning_results_file = open("tuning_results.txt", "w")

for dataset_name in classification_data.keys():
    tuning_results_file.write(f"{dataset_name}:\n")
    train_data, test_data = classification_data[dataset_name]
    result = tune.run(
        partial(tuning_train, train_data=train_data, test_data=test_data,
                criterion=binary_classification_criterion, alpha=alpha, epochs=epochs),
        resources_per_trial={"cpu": 8, "gpu": 1},
        config=config,
        num_samples=25,
        scheduler=scheduler
    )
    best_trial = result.get_best_trial("loss", "min")
    tuning_results_file.write(str(best_trial.config) + "\n\n")

for dataset_name in regression_data.keys():
    tuning_results_file.write(f"{dataset_name}:\n")
    train_data, test_data = regression_data[dataset_name]
    result = tune.run(
        partial(tuning_train, train_data=train_data, test_data=test_data,
                criterion=regression_criterion, alpha=alpha, epochs=epochs),
        resources_per_trial={"cpu": 8, "gpu": 1},
        config=config,
        num_samples=25,
        scheduler=scheduler
    )
    best_trial = result.get_best_trial("loss", "min")
    tuning_results_file.write(str(best_trial.config) + "\n\n")
