import pandas as pd
from torch.utils.data import Dataset

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pandas.api.types import is_numeric_dtype, is_float_dtype
import numpy as np


class BaseDataset(Dataset):
    def __init__(self):
        pass

    def _dtypes_to_float32(self, dataset):
        # float64 -> float32
        dataset[dataset.select_dtypes(np.float64).columns] = dataset.select_dtypes(
            np.float64
        ).astype(np.float32)
        # int64 -> float32
        dataset[dataset.select_dtypes(np.int64).columns] = dataset.select_dtypes(
            np.int64
        ).astype(np.float32)
        return dataset

    def _one_hot(self, dataset):
        """Performs one-hot encoding on categorical columns"""
        dataset_one_hot = dataset.copy()
        for key in dataset.keys():
            if not is_numeric_dtype(dataset[key]) and not is_float_dtype(dataset[key]):
                # Drop the column containing non numerical values and append ohe columns
                dataset_one_hot = dataset_one_hot.drop(key, axis=1)
                one_hot_from_column = pd.get_dummies(dataset[key], dtype=int)
                dataset_one_hot = pd.concat(
                    [dataset_one_hot, one_hot_from_column], axis=1
                )
        return dataset_one_hot

    def _normalize_features(self):
        """Normalizes the features to mean=0 and std=1"""
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


class CaliforniaHousingDataset(BaseDataset):
    """Loads and contains california housing dataset

    https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html

    Example usage:
        from torch.utils.data import DataLoader

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

    def __init__(
        self,
        dataset_path: str,
        normalize: bool = False,
        train: bool = True,
        train_size: float = 0.8,
    ):
        """Initializes dataset

        Args:
            dataset_path: Path to cal_housing.data file
            normalize: Boolean whether to normalize dataset to mean=0 and std=1
            train: Boolean whether to extract training set
            train_size: Fraction of the dataset devoted to the training set
        """
        dataset = pd.read_csv(dataset_path, header=None)
        dataset = self._dtypes_to_float32(dataset)

        self.target_col = dataset.columns[-1]  # 8, Median house value
        self.features, self.target = self._split_features_target(dataset)

        if normalize:
            self._normalize_features()

        features_train, features_test, target_train, target_test = train_test_split(
            self.features, self.target, train_size=train_size
        )
        if train:
            self.features = features_train
            self.target = target_train
        else:
            self.features = features_test
            self.target = target_test


class AdultDataset(BaseDataset):
    """Loads and contains the adult dataset

    https://archive.ics.uci.edu/dataset/2/adult

    Example usage:
        from torch.utils.data import DataLoader

        adult_path = "path/to/adult.data"

        test_dataset = AdultDataset(adult_path, train=False)
        train_dataset = AdultDataset(adult_path, train=True)

        test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    Attributes:
        dataset: pd.DataFrame. Whole dataset
        features: pd.DataFrame. Data containing features
        target: pd.DataFrame. Data containing targets
    """

    def __init__(
            self,
            dataset_path: str,
            normalize: bool = False,
            train: bool = True,
            train_size: float = 0.8
    ):
        """Initializes dataset

        Args:
            dataset_path (str): Path to adult.data file
            normalize: Boolean whether to normalize dataset to mean=0 and std=1
            train: Boolean whether to extract training set
            train_size: Fraction of the dataset devoted to the training set
        """
        dataset = self._load_and_preprocess_data(dataset_path)
        self.target_col = dataset.columns[-1]
        self.features, self.target = self._split_features_target(dataset)

        if normalize:
            self._normalize_features()

        features_train, features_test, target_train, target_test = train_test_split(
            self.features, self.target, train_size=train_size
        )
        if train:
            self.features = features_train
            self.target = target_train
        else:
            self.features = features_test
            self.target = target_test

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
        dataset = self._one_hot(dataset).drop(" <=50K", axis=1)

        dataset = self._dtypes_to_float32(dataset)
        return dataset


class WineDataset(BaseDataset):
    """Loads and contains wine quality dataset

    https://archive.ics.uci.edu/dataset/186/wine+quality

    Example usage:
        from torch.utils.data import DataLoader

        wine_path = "path/to/winequality-red.csv"
        # wine_path = "path/to/winequality-white.csv"

        test_dataset = WineDataset(wine_path, normalize=True, train=False)
        train_dataset = WineDataset(wine_path, normalize=True, train=True)

        test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    Attributes:
        target_col: int. Index of target column in the dataset
        dataset: pd.DataFrame. Whole dataset
        features: pd.DataFrame. Data containing features
        target: pd.DataFrame. Data containing targets
    """

    def __init__(
        self,
        dataset_path: str,
        normalize: bool = False,
        train: bool = True,
        train_size: float = 0.8,
    ):
        """Initializes dataset

        Args:
            dataset_path: Path to winequality-red.csv or winequality-white.csv file
            normalize: Boolean whether to normalize dataset to mean=0 and std=1
            train: Boolean whether to extract training set
            train_size: Fraction of the dataset devoted to the training set
        """
        dataset = pd.read_csv(dataset_path, sep=";")
        dataset = self._dtypes_to_float32(dataset)

        self.target_col = dataset.columns[-1]  # 12, wine quality
        self.features, self.target = self._split_features_target(dataset)

        if normalize:
            self._normalize_features()
        features_train, features_test, target_train, target_test = train_test_split(
            self.features, self.target, train_size=train_size
        )
        if train:
            self.features = features_train
            self.target = target_train
        else:
            self.features = features_test
            self.target = target_test


class AutoMpgDataset(BaseDataset):
    """Loads and contains auto mpg dataset

    http://archive.ics.uci.edu/dataset/9/auto+mpg

    Example usage:
        from torch.utils.data import DataLoader

        auto_path = "path/to/auto-mpg.data"

        test_dataset =  AutoMpgDataset(auto_path, normalize=True, train=False)
        train_dataset = AutoMpgDataset(auto_path, normalize=True, train=True)

        test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    Attributes:
        target_col: int. Index of target column in the dataset
        dataset: pd.DataFrame. Whole dataset
        features: pd.DataFrame. Data containing features
        target: pd.DataFrame. Data containing targets
    """

    def __init__(
        self,
        dataset_path: str,
        normalize: bool = False,
        train: bool = True,
        train_size: float = 0.8,
    ):
        """Initializes dataset

        Args:
            dataset_path: Path to auto-mpg.data
            normalize: Boolean whether to normalize dataset to mean=0 and std=1
            train: Boolean whether to extract training set
            train_size: Fraction of the dataset devoted to the training set
        """
        dataset = pd.read_csv(dataset_path, header=None, delim_whitespace=True)
        dataset = self._preprocess_dataset(dataset)
        self.target_col = dataset.columns[0]  # 0, mpg - miles per galon
        self.features, self.target = self._split_features_target(dataset)

        if normalize:
            self._normalize_features()
        features_train, features_test, target_train, target_test = train_test_split(
            self.features, self.target, train_size=train_size
        )
        if train:
            self.features = features_train
            self.target = target_train
        else:
            self.features = features_test
            self.target = target_test

    def _preprocess_dataset(self, dataset):
        """Drops car model name column and deletes rows that contain '?' value"""
        dataset = dataset.drop(
            dataset.columns[-1], axis=1
        )  # Drop column with car model name (unique string)

        dataset.dropna()
        for col in dataset.columns:
            unique_values = dataset[col].unique()
            if "?" in unique_values:
                col_name = col

        dataset = dataset[dataset[col_name] != "?"]

        # 3rd column has dtype object
        # we have to convert exclusively
        dataset[3] = dataset[3].astype(np.float32)

        dataset = self._dtypes_to_float32(dataset)

        return dataset


class TitanicDataset(BaseDataset):
    """Loads and contains titanic dataset

    https://www.openml.org/search?type=data&sort=runs&id=40945&status=active

    Example usage:
        from torch.utils.data import DataLoader

        titanic_path = "path/to/titanic.arff"

        test_dataset = TitanicDataset(titanic_path, normalize=True, train=False)
        train_dataset =  TitanicDataset(titanic_path, normalize=True, train=True)

        test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    Attributes:
        target_col: int. Index of target column in the dataset
        dataset: pd.DataFrame. Whole dataset
        features: pd.DataFrame. Data containing features
        target: pd.DataFrame. Data containing targets
    """

    def __init__(
        self,
        dataset_path: str,
        normalize: bool = False,
        train: bool = True,
        train_size: float = 0.8,
    ):
        """Initializes dataset

        Args:
            dataset_path: Path to titanic.arff
            normalize: Boolean whether to normalize dataset to mean=0 and std=1
            train: Boolean whether to extract training set
            train_size: Fraction of the dataset devoted to the training set

        0 @attribute 'pclass' numeric
        1 @attribute 'survived' {0,1}
        2 @attribute 'name' string
        3 @attribute 'sex' {'female','male'}
        4 @attribute 'age' numeric
        5 @attribute 'sibsp' numeric
        6 @attribute 'parch' numeric
        7 @attribute 'ticket' string
        8 @attribute 'fare' numeric
        9 @attribute 'cabin' string
        10 @attribute 'embarked' {'C','Q','S'}
        11 @attribute 'boat' string
        12 @attribute 'body' numeric
        13 @attribute 'home.dest' string
        """
        dataset = pd.read_csv(dataset_path, skiprows=17, header=None)
        dataset = self._preprocess_dataset(dataset)

        self.target_col = dataset.columns[1]  # survived {0, 1}
        self.features, self.target = self._split_features_target(dataset)
        self.features = self.features.rename(str, axis="columns")

        if normalize:
            self._normalize_features()
        features_train, features_test, target_train, target_test = train_test_split(
            self.features, self.target, train_size=train_size
        )
        if train:
            self.features = features_train
            self.target = target_train
        else:
            self.features = features_test
            self.target = target_test

    def _preprocess_dataset(self, dataset):
        # Drop 2 name, 7 ticket, 9 cabin, 11 boat, 12 body, 13 home dest column
        to_drop = dataset.columns[[2, 7, 9, 11, 12, 13]]
        dataset = dataset.drop(to_drop, axis=1)
        dataset.dropna()

        col_names = [col for col in dataset.columns if "?" in dataset[col].unique()]

        for col in col_names:
            dataset = dataset[dataset[col] != "?"]

        dataset[4] = pd.to_numeric(dataset[4])  # Convert age to numeric
        dataset[8] = pd.to_numeric(dataset[8])  # Convert fare to numeric

        dataset = self._one_hot(dataset)
        dataset = self._dtypes_to_float32(dataset)

        return dataset
