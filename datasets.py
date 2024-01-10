import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pandas.api.types import is_numeric_dtype


class CaliforniaHousingDataset(Dataset):
    """Loads and contains california housing dataset

    https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html

    Example usage:
        from torch.utils.data import DataLoader, random_split

        california_housing_path = "path/to/cal_housing.data"

        test_dataset = CaliforniaHousingDataset(california_housing_path, normalize=True, train=False)
        train_dataset = CaliforniaHousingDataset(california_housing_path, normalize=True, train=True)

        test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    Attributes:
        target_col: int. Index of target column in the dataset
        dataset: pd.DataFrame. Whole dataset
        features: pd.DataFrame. Data containing features
        target: pd.DataFrame. Data containing targets
    """

    def __init__(self, dataset_path: str, normalize: bool = False, train: bool = True, train_size: float = 0.8):
        """Initializes dataset

        Args:
            dataset_path: Path to cal_housing.data file
            normalize: Boolean whether to normalize dataset to mean=0 and std=1
            train: Boolean whether to extract training set
            train_size: Fraction of the dataset devoted to the training set
        """
        dataset = pd.read_csv(dataset_path, header=None)
        self.target_col = dataset.columns[-1]  # 8, Median house value
        self.features, self.target = self._split_features_target(dataset)

        if normalize:
            self._normalize_dataset()

        features_train, features_test, target_train, target_test = train_test_split(
            self.features, self.target, train_size=train_size)
        if train:
            self.features = features_train
            self.target = target_train
        else:
            self.features = features_test
            self.target = target_test

    def _normalize_dataset(self):
        """Normalizes the dataset to mean=0 and std=1"""
        scaler = StandardScaler()
        self.features = pd.DataFrame(scaler.fit_transform(self.features))

    def _split_features_target(self, dataset):
        """Splits the dataset into features and target"""
        features = dataset.drop(self.target_col, axis=1)
        target = dataset[self.target_col]
        return features, target

    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.features)

    def __getitem__(self, idx):
        """Returns the features and target for a given index"""
        features = self.features.iloc[idx].values
        target = self.target.iloc[idx]
        return features, target


class AdultDataset(Dataset):
    """Loads and contains the adult dataset

    https://archive.ics.uci.edu/dataset/2/adult

    Example usage:
        from torch.utils.data import DataLoader, random_split

        path = "path/to/adult.data"

        dataset = CaliforniaHousingDataset(path)
        test_dataset, train_dataset = random_split(dataset, [0.2, 0.8])

        test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    Attributes:
        dataset: pd.DataFrame. Whole dataset
        features: pd.DataFrame. Data containing features
        target: pd.DataFrame. Data containing targets
    """

    def __init__(self, dataset_path: str):
        """Initializes the dataset

        Args:
            dataset_path (str): Path to adult.data file
        """
        self.dataset = self._load_and_preprocess_data(dataset_path)
        self.features, self.target = self._split_features_target()

    def _load_and_preprocess_data(self, dataset_path):
        """Loads and preprocesses the dataset"""
        dataset = pd.read_csv(dataset_path, header=None)
        # Drop the 3rd column "education" as it's redundant
        dataset = dataset.drop(3, axis=1).dropna()

        # Find columns that have " ?" as values
        columns_with_question_mark_as_values = [
            key for key in dataset.keys() if " ?" in dataset[key].unique()
        ]

        # Replace values " ?" with "{column_name}_unknown"
        for column in columns_with_question_mark_as_values:
            dataset[column] = dataset[column].replace([" ?"], f"{column}_unknown")

        # After ohe, one of the columns is redundant (" <=50K", ">50K")
        return self._one_hot(dataset).drop(" <=50K", axis=1)

    def _one_hot(self, dataset):
        """Performs one-hot encoding on categorical columns"""
        # Copy dataset
        dataset_one_hot = dataset
        for key in dataset.keys():
            if not is_numeric_dtype(dataset[key]):
                # Drop the column containing non numerical values and append ohe columns
                dataset_one_hot = dataset_one_hot.drop(key, axis=1)
                one_hot_from_column = pd.get_dummies(dataset[key], dtype=int)
                dataset_one_hot = pd.concat(
                    [dataset_one_hot, one_hot_from_column], axis=1
                )
        return dataset_one_hot

    def _split_features_target(self):
        """Splits the dataset into features and target"""
        features = self.dataset.drop(self.dataset.columns[-1], axis=1)
        target = self.dataset[self.dataset.columns[-1]]
        return features, target

    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.dataset)

    def __getitem__(self, idx):
        """Returns the features and target for a given index"""
        features = self.features.iloc[idx].values
        target = self.target.iloc[idx]
        return features, target
