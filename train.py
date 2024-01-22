import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import sklearn.metrics
from early_stopping import EarlyStopping
from log_cosh_loss import LogCoshLoss

from datasets import CaliforniaHousingDataset, AdultDataset, TitanicDataset, AutoMpgDataset, WineDataset
from metrics import calculate_global_fidelity, calculate_global_neighborhood_fidelity
from models.base_model import BaseClassifier, BaseRegressor
from models.surrogate_model import SurrogateClassifier, SurrogateRegressor


def load_housing_data():
    housing_train = CaliforniaHousingDataset(
        dataset_path="data/california_housing/cal_housing.data", standardize=True, train=True)
    housing_test = CaliforniaHousingDataset(
        dataset_path="data/california_housing/cal_housing.data", standardize=True, train=False)
    return housing_train, housing_test


def load_adult_data():
    adult_train = AdultDataset(dataset_path="data/adult/adult.data", standardize=True, train=True)
    adult_test = AdultDataset(dataset_path="data/adult/adult.data", standardize=True, train=False)
    return adult_train, adult_test


def load_wine_data():
    wine_train = WineDataset(dataset_path="data/wines/winequality-red.csv", standardize=True, train=True)
    wine_test = WineDataset(dataset_path="data/wines/winequality-red.csv", standardize=True, train=False)
    return wine_train, wine_test


def load_titanic_data():
    titanic_train = TitanicDataset(dataset_path="data/titanic/titanic.arff", standardize=True, train=True)
    titanic_test = TitanicDataset(dataset_path="data/titanic/titanic.arff", standardize=True, train=False)
    return titanic_train, titanic_test


def load_autompg_data():
    autompg_train = AutoMpgDataset(dataset_path="data/autompg/auto-mpg.data", standardize=True, train=True)
    autompg_test = AutoMpgDataset(dataset_path="data/autompg/auto-mpg.data", standardize=True, train=False)
    return autompg_train, autompg_test


def create_regressors(
        dataset: Dataset,
        n_hidden_layers: int,
        layer_size: int
):
    base_model = BaseRegressor(dataset.features.shape[1], 1, n_hidden_layers, layer_size).to(device)
    surrogate_model = SurrogateRegressor(dataset.features.shape[1], 1).to(device)
    return base_model, surrogate_model


def create_classifiers(
        dataset: Dataset,
        n_hidden_layers: int,
        layer_size: int
):
    base_model = BaseClassifier(dataset.features.shape[1], 1, n_hidden_layers, layer_size).to(device)
    surrogate_model = SurrogateClassifier(dataset.features.shape[1], 1).to(device)
    return base_model, surrogate_model


def train(
        base_model: nn.Module,
        surrogate_model: nn.Module,
        train_data: Dataset,
        batch_size: int,
        criterion,
        epochs: int,
        alpha: float,
        early_stopping: EarlyStopping = None
):
    params = list(base_model.parameters()) + list(surrogate_model.parameters())
    optimizer = Adam(params, lr=lr)
    loader = DataLoader(train_data, batch_size=batch_size)
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

        train_loss = running_loss / len(loader)
        if early_stopping is not None:
            early_stopping(train_loss, base_model, surrogate_model)
            if early_stopping.early_stop:
                #print("Early stopping")
                break
        #print(f"epoch: {epoch + 1}, train loss: {train_loss:.3f}")


def validate_base_classifier(
        model: nn.Module,
        test_data: Dataset,
):
    loader = DataLoader(test_data, batch_size=len(test_data))
    with torch.no_grad():
        data, labels = next(iter(loader))
        data, labels = data.to(device), labels.to(device)
        labels = labels.reshape(-1, 1)
        preds_proba = model(data)
        preds = torch.where(preds_proba >= 0.5, 1, 0)
        accuracy = sklearn.metrics.accuracy_score(labels.cpu(), preds.cpu())
        f1_score = sklearn.metrics.f1_score(labels.cpu(), preds.cpu())
        print(f"test accuracy: {accuracy:.3f}, f1 score: {f1_score:.3f}")


def validate_base_regressor(
        model: nn.Module,
        test_data: Dataset
):
    loader = DataLoader(test_data, batch_size=len(test_data))
    with torch.no_grad():
        data, labels = next(iter(loader))
        data, labels = data.to(device), labels.to(device)
        labels = labels.reshape(-1, 1)
        preds = model(data)
        mse = sklearn.metrics.mean_squared_error(labels.cpu(), preds.cpu())
        print(f"test mse: {mse:.3f}")


def validate_surrogate_model(
        base_model: nn.Module,
        surrogate_model: nn.Module,
        test_data: Dataset
):
    loader = DataLoader(test_data, batch_size=len(test_data))
    with torch.no_grad():
        data, _ = next(iter(loader))
        data = data.to(device)
        base_model_preds = base_model(data)
        surrogate_model_preds = surrogate_model(data)
        global_fidelity = calculate_global_fidelity(base_model_preds, surrogate_model_preds)
        print(
            f"global fidelity: {global_fidelity:.3f}")


def validate_regressors(
        base_model: nn.Module,
        surrogate_model: nn.Module,
        test_data: Dataset
):
    validate_base_regressor(base_model, test_data)
    validate_surrogate_model(base_model, surrogate_model, test_data)
    print()


def validate_classifiers(
        base_model: nn.Module,
        surrogate_model: nn.Module,
        test_data: Dataset
):
    validate_base_classifier(base_model, test_data)
    validate_surrogate_model(base_model, surrogate_model, test_data)
    print()


device = "cuda:0" if torch.cuda.is_available() else "cpu"

lr = 0.001
binary_classification_criterion = torch.nn.BCELoss()
regression_criterion = LogCoshLoss()  # "logarithm of the hyperbolic cosine" from the paper

epochs = 999
patience = 5  # for early stopping
save_dir = "checkpoints"

adult_train, adult_test = load_adult_data()
titanic_train, titanic_test = load_titanic_data()
wine_train, wine_test = load_wine_data()
housing_train, housing_test = load_housing_data()
autompg_train, autompg_test = load_autompg_data()
classification_data = {
    "adult": (adult_train, adult_test),
    "titanic": (titanic_train, titanic_test)
}
regression_data = {
    "wine": (wine_train, wine_test),
    "housing": (housing_train, housing_test),
    "autompg": (autompg_train, autompg_test)
}

# hyperparameters as in tuning_results.txt
adult_n_layers, adult_layer_size, adult_batch_size = 5, 64, 64
titanic_n_layers, titanic_layer_size, titanic_batch_size = 4, 16, 32
wine_n_layers, wine_layer_size, wine_batch_size = 3, 128, 32
housing_n_layers, housing_layer_size, housing_batch_size = 3, 256, 128
autompg_n_layers, autompg_layer_size, autompg_batch_size = 6, 512, 64

classifier_hyperparameters = {
    "adult": (adult_n_layers, adult_layer_size, adult_batch_size),
    "titanic": (titanic_n_layers, titanic_layer_size, titanic_batch_size),
}
regressor_hyperparameters = {
    "wine": (wine_n_layers, wine_layer_size, wine_batch_size),
    "housing": (housing_n_layers, housing_layer_size, housing_batch_size),
    "autompg": (autompg_n_layers, autompg_layer_size, autompg_batch_size)
}

for dataset_name in classification_data.keys():
    train_data, test_data = classification_data[dataset_name]
    print(f"{dataset_name}:")
    for alpha in [0.1 * i for i in range(1, 10)]:
        n_layers, layer_size, batch_size = classifier_hyperparameters[dataset_name]
        base_model, surrogate_model = create_classifiers(train_data, n_layers, layer_size)
        early_stopping = EarlyStopping(dir=save_dir, dataset_name=dataset_name, patience=patience, verbose=False)
        train(base_model, surrogate_model, train_data, batch_size,
              binary_classification_criterion, epochs, alpha, early_stopping)
        print(alpha)
        validate_classifiers(base_model, surrogate_model, test_data)

for dataset_name in regression_data.keys():
    train_data, test_data = regression_data[dataset_name]
    print(f"{dataset_name}:")
    for alpha in [0.1 * i for i in range(1, 10)]:
        n_layers, layer_size, batch_size = regressor_hyperparameters[dataset_name]
        base_model, surrogate_model = create_regressors(train_data, n_layers, layer_size)
        early_stopping = EarlyStopping(dir=save_dir, dataset_name=dataset_name, patience=patience, verbose=False)
        train(base_model, surrogate_model, train_data, batch_size,
              regression_criterion, epochs, alpha, early_stopping)
        print(alpha)
        validate_regressors(base_model, surrogate_model, test_data)

