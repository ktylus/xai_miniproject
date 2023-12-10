import os
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class CaliforniaHousingDataset(Dataset):
    """Loads and contains california housing dataset
    
    Example usage:
        from torch.utils.data import DataLoader, random_split

        california_housing_path = "path/to/cal_housing.data"

        dataset = CaliforniaHousingDataset(california_housing_path, normalize=True)
        test_dataset, train_dataset = random_split(dataset, [0.2, 0.8])

        test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    Attributes:
        target_col: int. Index of target column in the dataset
        dataset: pd.DataFrame. Whole dataset
        features: pd.DataFrame. Data containing features
        target: pd.DataFrame. Data containing targets
    """
    def __init__(self, dataset_path: str, normalize: bool=False):
        """Initializes dataset

        Args:
            dataset_path: Path to cal_housing.data file
            normalize: Boolean whether to normalize dataset to mean=0 and std=1
        """
        self.dataset = pd.read_csv(dataset_path, header=None)

        if normalize:
            self.normalize_dataset()

        self.target_col = 8  # Median house value
        self.features, self.target = self.split_features_target()

    def normalize_dataset(self):
        scaler = StandardScaler()
        self.dataset = pd.DataFrame(scaler.fit_transform(self.dataset))

    def split_features_target(self):
        features = self.dataset.drop(self.target_col, axis=1)
        target = self.dataset[self.target_col]
        return features, target

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        features = self.features.iloc[idx].values
        target = self.target.iloc[idx]
        return features, target
