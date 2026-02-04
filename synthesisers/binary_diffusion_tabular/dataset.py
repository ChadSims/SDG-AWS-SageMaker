from typing import List, Tuple, Union, Optional, Dict

import pandas as pd

import torch
from torch.utils.data import Dataset

from synthesisers.binary_diffusion_tabular.transformation import (
    FixedSizeBinaryTableTransformation,
    TASK,
)


__all__ = ["FixedSizeBinaryTableDataset", "drop_fill_na"]


def drop_fill_na(
    df: pd.DataFrame,
    columns_numerical: List[str],
    columns_categorical: List[str],
    dropna: bool,
    fillna: bool,
) -> pd.DataFrame:
    """Drops or fills NaN values in a dataframe

    Args:
        df: dataframe
        columns_numerical: numerical column names
        columns_categorical:  categorical column names
        dropna: if True, drops NaN values
        fillna:  if True, fills NaN values. Numerical columns are replaced with mean. Categorical columns are replaced
                 with mode.

    Returns:
        pd.DataFrame: dataframe with NaN values dropped/filled
    """

    if dropna and fillna:
        raise ValueError("Cannot have both dropna and fillna")

    if dropna:
        df = df.dropna()

    if fillna:
        for col in columns_numerical:
            df[col] = df[col].fillna(df[col].mean())

        # replace na for categorical columns with mode
        for col in columns_categorical:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


class FixedSizeBinaryTableDataset(Dataset):
    """Pytorch dataset for fixed size binary tables."""

    def __init__(
        self,
        *,
        table: pd.DataFrame,
        target_column: Optional[str] = None,
        split_feature_target: bool,
        task: TASK,
        numerical_columns: List[str] = None,
        categorical_columns: List[str] = None,
    ):
        """
        Args:
            table: pandas dataframe with categorical and numerical columns. Dataframe should not have nan
            target_column: name of the target column. Optional. Should be provided if split_feature_target is True.
            split_feature_target: split features columns and target column
            task: task for which dataset is used. Can be 'classification' or 'regression'
            numerical_columns: list of columns with numerical values
            categorical_columns: list of columns with categorical values
        """

        if numerical_columns is None:
            numerical_columns = []

        if categorical_columns is None:
            categorical_columns = []

        self.table = table
        self.target_column = target_column
        self.split_feature_target = split_feature_target
        self.task = task
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns

        self.transformation = FixedSizeBinaryTableTransformation(
            task, numerical_columns, categorical_columns
        )

        if self.split_feature_target:
            target = self.table[self.target_column]
            features = self.table.drop(columns=[self.target_column])

            self.features_binary, self.targets_binary = (
                self.transformation.fit_transform(features, target)
            )
        else:
            self.features_binary = self.transformation.fit_transform(self.table)

    @classmethod
    def from_config(cls, config: Dict) -> "FixedSizeBinaryTableDataset":
        path_table = config["path_table"]
        df = pd.read_csv(path_table)
        dropna = config["dropna"]
        fillna = config["fillna"]
        columns_numerical = config["numerical_columns"]
        columns_categorical = config["categorical_columns"]
        columns_to_drop = config["columns_to_drop"]
        task = config["task"]

        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
        df = drop_fill_na(df, columns_numerical, columns_categorical, dropna, fillna)

        return cls(
            table=df,
            target_column=config["target_column"],
            task=task,
            split_feature_target=config["split_feature_target"],
            numerical_columns=config["numerical_columns"],
            categorical_columns=config["categorical_columns"],
        )

    @property
    def n_classes(self) -> int:
        return self.transformation.n_classes if self.split_feature_target else 0

    @property
    def row_size(self) -> int:
        return self.transformation.row_size

    @property
    def conditional(self) -> bool:
        return self.split_feature_target

    def __len__(self) -> int:
        return len(self.features_binary)

    def __getitem__(
        self, idx: int
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        row = self.features_binary[idx]

        if self.split_feature_target:
            target = self.targets_binary[idx]
            return row, target
        return row
