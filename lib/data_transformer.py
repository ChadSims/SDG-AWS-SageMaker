import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

class DataTransformer(object):
    def __init__(self):
        self.num_features = []
        self.cat_features = []
        self.label = None
        self.cat_transformers = {}
        self.label_transformer = None

        self.is_fitted = False

    def fit(self, data: pd.DataFrame, cat_features: list, label: str, task:str):  

        self.is_fitted = True
        self.label = label
        self.num_features = [col for col in data.columns if col not in cat_features and col != label]
        self.cat_features = cat_features

        if self.cat_features:
            self.cat_transformers = {
                col: OrdinalEncoder(
                    handle_unknown='use_encoded_value',
                    unknown_value=-1
                ) 
                for col in self.cat_features
            }

            for col in self.cat_features:
                # data[col] = data[col].astype(str)
                self.cat_transformers[col].fit(data[col].values.reshape(-1, 1))

        if task == "classification":
            self.label_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            self.label_transformer.fit(data[label].values.reshape(-1, 1))


    def transform(self, data: pd.DataFrame) -> pd.DataFrame:

        if not self.is_fitted:
            raise RuntimeError("DataTransformer is not fitted yet. Please call fit() first.")
            
        num_data = data[self.num_features].values

        if self.cat_features:
            cat_data = np.concatenate(
                [
                    self.cat_transformers[col].transform(data[col].values.reshape(-1, 1))
                    for col in self.cat_features
                ],
                axis=1
            )

        if self.label_transformer:
            label_data = self.label_transformer.transform(data[self.label].values.reshape(-1, 1)).reshape(-1, 1)
        else:
            label_data = data[self.label].values.reshape(-1, 1)

        if self.cat_features:
            # concatenate transformed data into numpy
            transformed_data = np.concatenate([num_data, cat_data, label_data], axis=1)
            transformed_df = pd.DataFrame(transformed_data, columns=self.num_features + self.cat_features + [self.label])

        else:
            transformed_data = np.concatenate([num_data, label_data], axis=1)
            transformed_df = pd.DataFrame(transformed_data, columns=self.num_features + [self.label])

        return transformed_df


    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:

        if not self.is_fitted:
            raise RuntimeError("DataTransformer is not fitted yet. Please call fit() first.")

        num_data = data[self.num_features].values

        if self.cat_features:
            cat_data = np.concatenate(
                [
                    self.cat_transformers[col].inverse_transform(data[col].values.reshape(-1, 1))
                    for col in self.cat_features
                ],
                axis=1
            )

        if self.label_transformer:
            if data[self.label].dtype != int: 
                data[self.label] = data[self.label].astype(int)

            label_data = self.label_transformer.inverse_transform(data[self.label].values.reshape(-1, 1)).reshape(-1, 1)
        else:
            label_data = data[self.label].values.reshape(-1, 1)

        if self.cat_features:
            # concatenate inverse transformed data into pandas DataFrame
            inverse_transformed_df = pd.DataFrame(
                np.concatenate([num_data, cat_data, label_data], axis=1),
                columns=self.num_features + self.cat_features + [self.label],
            )
        else:
            inverse_transformed_df = pd.DataFrame(
                np.concatenate([num_data, label_data], axis=1),
                columns=self.num_features + [self.label],
            )

        return inverse_transformed_df


    def fit_transform(self, data: pd.DataFrame, cat_features: list, label: str, task: str):
        self.fit(data, cat_features, label, task)
        return self.transform(data)
