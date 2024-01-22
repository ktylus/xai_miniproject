import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import sklearn.metrics
from sklearn.preprocessing import StandardScaler
from early_stopping import EarlyStopping
from log_cosh_loss import LogCoshLoss
from lime import lime_tabular
from datasets import (
    CaliforniaHousingDataset,
    AdultDataset,
    TitanicDataset,
    AutoMpgDataset,
    WineDataset,
)
from metrics import calculate_global_fidelity, calculate_global_neighborhood_fidelity_lime
from models.base_model import BaseClassifier, BaseRegressor
from models.surrogate_model import SurrogateClassifier, SurrogateRegressor

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def predict_proba(samples):
    samples_tensor = torch.tensor(samples, dtype=torch.float32).to(device)
    predictions = base_model(samples_tensor).cpu().detach().numpy()
    return np.column_stack((1 - predictions, predictions))

def load_adult_data():
    adult_train = AdultDataset(dataset_path="data/adult/adult.data", normalize=True, train=True)
    adult_test = AdultDataset(dataset_path="data/adult/adult.data", normalize=True, train=False)
    return adult_train, adult_test


def load_titanic_data():
    titanic_train = TitanicDataset(dataset_path="data/titanic/titanic.arff", normalize=True, train=True)
    titanic_test = TitanicDataset(dataset_path="data/titanic/titanic.arff", normalize=True, train=False)
    return titanic_train, titanic_test

classification_data = {
    "adult": load_adult_data(),
    "titanic": load_titanic_data()
}
save_dir = "checkpoints"

for i, dataset in enumerate(classification_data.keys()):
    train_data = classification_data[dataset][0]
    test_data = classification_data[dataset][1]
    
    base_model = BaseClassifier(
        input_dim=test_data.features.shape[1], output_dim=1, n_hidden_layers=4, layer_size=128).to(device)
    
    base_model.load_state_dict(torch.load(f"{save_dir}/{dataset}/base_model_checkpoint.pt"))
    base_model.eval()
    
    explainer = lime_tabular.LimeTabularExplainer(
    train_data.features.values,
    mode="classification",
    feature_names=train_data.features.columns,
    categorical_features=None,
    sample_around_instance=True,
    )
    
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    gnf = 0
    for i, (sample, _) in enumerate(test_loader):
        print(i)
        sample = sample.to(device)
        exp, linear_model = explainer.explain_instance(
            sample.cpu().numpy()[0], predict_proba, num_features=15)
        gnf += calculate_global_neighborhood_fidelity_lime(base_model, linear_model, sample)

    result = gnf / len(test_data)
    print(f"{dataset} GNF: {result}")