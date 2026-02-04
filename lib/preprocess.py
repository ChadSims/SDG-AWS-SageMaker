import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer


def clean(df: pd.DataFrame):
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

def split(df, target_column, task, test_size=0.2, random_state=42):
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    if task == 'regression':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, random_state=random_state)
    else: # classification
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, stratify=y_train, random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test

def transform(X_train, X_val, X_test, y_train, y_val, y_test, num_features, cat_features, task):

    feature_transformer = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ],
        remainder='drop'
    )

    if task == 'classification':
        target_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    else:
        target_transformer = StandardScaler()
        y_train = y_train.values.reshape(-1, 1)
        y_val = y_val.values.reshape(-1, 1)
        y_test = y_test.values.reshape(-1, 1)

    X_train = feature_transformer.fit_transform(X_train)
    X_val = feature_transformer.transform(X_val)
    X_test = feature_transformer.transform(X_test)
    if isinstance(y_train, pd.Series):
        y_train = target_transformer.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        y_val = target_transformer.transform(y_val.values.reshape(-1, 1)).ravel()
        y_test = target_transformer.transform(y_test.values.reshape(-1, 1)).ravel()
    elif isinstance(y_train, np.ndarray):
        y_train = target_transformer.fit_transform(y_train).ravel()
        y_val = target_transformer.transform(y_val).ravel()
        y_test = target_transformer.transform(y_test).ravel()
    

    return X_train, X_val, X_test, y_train, y_val, y_test

def preprocess(df, metadata, test_size=0.2, random_state=42):

    num_features = metadata['num_features']
    cat_features = metadata['cat_features']
    target_column = metadata['target_column']
    task = metadata['task']
    
    clean(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split(df, target_column, task, test_size, random_state)
    X_train, X_val, X_test, y_train, y_val, y_test = transform(X_train, X_val, X_test, y_train, y_val, y_test,
                                                               num_features, cat_features, task)
    return X_train, X_val, X_test, y_train, y_val, y_test

def normalise(df: pd.DataFrame, num_features, scaler='standard'):
    """
    Normalize numerical features in the DataFrame.
    scaler can be 'standard' or 'minmax'.
    """
    if scaler == 'minmax':
        scaler = MinMaxScaler()
    elif scaler == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("Scaler must be 'minmax' or 'standard'.")
    
    cat_features = [col for col in df.columns if col not in num_features]
    
    column_transformer = ColumnTransformer(
        [
            ('num', scaler, num_features)
        ],
        remainder='passthrough'  # Keep non-numerical features unchanged
    )

    transformed = column_transformer.fit_transform(df)
    transformed_df = pd.DataFrame(transformed, columns=num_features + cat_features)