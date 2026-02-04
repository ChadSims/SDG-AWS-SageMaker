import argparse
import os 
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split

from lib.imputer import impute
from lib.utils import load_dataset, load_config, dump_config
# from lib.info import DATASETS_PATH

DATASETS_PATH = os.getenv("DATASETS_PATH", "/opt/ml/input/data/datasets")


def main():
    parser = argparse.ArgumentParser(description="Impute missing values in a dataset.")
    parser.add_argument('--dataset', type=str, required=True, help='The name of the dataset (without extension).')

    args = parser.parse_args()

    dataset = args.dataset

    dataset_path = os.path.join(DATASETS_PATH, f"{dataset}/{dataset}.csv")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}. Please provide a valid dataset path.")
    
    metadata_path = os.path.join(DATASETS_PATH, f"{dataset}/metadata.toml")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}. Please generate metadata first.")
    
    metadata = load_config(metadata_path)
    target_column = metadata['target_column']
    task = metadata['task']

    df = pd.read_csv(dataset_path)

    df.drop_duplicates(inplace=True)

    test_index_path = os.path.join(DATASETS_PATH, f"{dataset}/test_index.pkl")
    if not os.path.exists(test_index_path):
        raise FileNotFoundError(f"Test index file not found at {test_index_path}. Please generate the test index first.")
    
    with open(test_index_path, 'rb') as f:
        test_index = pickle.load(f)

    # only train and val as that is what was used to train synthesiser
    real_data = df.drop(index=test_index)

    real_data.reset_index(drop=True, inplace=True)

    # impute
    imputed_data = impute(real_data, metadata)

    imputed_data_path = os.path.join(DATASETS_PATH, f"{dataset}_imputed/{dataset}_imputed.csv")
    os.makedirs(os.path.dirname(imputed_data_path), exist_ok=True)

    imputed_data.to_csv(imputed_data_path, index=False)

    X = imputed_data.drop(target_column, axis=1)
    y = imputed_data[target_column]

    if task == 'regression':
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    else: # classification
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    train = pd.concat([X_train, y_train], axis=1)
    val = pd.concat([X_val, y_val], axis=1)

    train.to_csv(os.path.join(DATASETS_PATH, f"{dataset}_imputed/train.csv"), index=False)
    val.to_csv(os.path.join(DATASETS_PATH, f"{dataset}_imputed/val.csv"), index=False)

    # copy the test set from the real dataset folder to the imputed dataset folder
    test = pd.read_csv(os.path.join(DATASETS_PATH, f"{dataset}/test.csv"))
    test.to_csv(os.path.join(DATASETS_PATH, f"{dataset}_imputed/test.csv"), index=False)

    dump_config(metadata, os.path.join(DATASETS_PATH, f"{dataset}_imputed/metadata.toml"))


if __name__ == "__main__":
    main()

