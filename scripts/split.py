import argparse
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

from lib.preprocess import clean, split
from lib.utils import load_config, dump_config

DATASETS_PATH = os.getenv("DATASETS_PATH", "/opt/ml/input/data/datasets")

def main():
    parser = argparse.ArgumentParser(description="Split dataset into train, validation, and test sets.")
    parser.add_argument('--dataset', type=str, required=True, help='The name of the dataset (without extension).')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of the dataset to include in the test split.')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for reproducibility.')

    args = parser.parse_args()

    dataset = args.dataset
    test_size = args.test_size
    random_state = args.random_state

    dataset_path = os.path.join(DATASETS_PATH, f"{dataset}/{dataset}.csv")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}. Please provide a valid dataset path.")

    metadata_path = os.path.join(DATASETS_PATH, f"{dataset}/metadata.toml")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}. Please generate metadata first.")
    
    df = pd.read_csv(dataset_path)

    metadata = load_config(metadata_path)
    target_column = metadata['target_column']
    task = metadata['task']

    metadata['test_size'] = test_size
    metadata['random_state'] = random_state

    dump_config(metadata, metadata_path)

    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    X_train, X_val, X_test, y_train, y_val, y_test = split(df, target_column, task, test_size=test_size, random_state=random_state)
    
    train = pd.concat([X_train, y_train], axis=1)
    val = pd.concat([X_val, y_val], axis=1)
    test = pd.concat([X_test, y_test], axis=1)

    test_index = test.index.tolist()

    with open(os.path.join(DATASETS_PATH, f"{dataset}/test_index.pkl"), 'wb') as f:
        pickle.dump(test_index, f)

    train.to_csv(os.path.join(DATASETS_PATH, f"{dataset}/train.csv"), index=False)
    val.to_csv(os.path.join(DATASETS_PATH, f"{dataset}/val.csv"), index=False)
    test.to_csv(os.path.join(DATASETS_PATH, f"{dataset}/test.csv"), index=False)

if __name__ == "__main__":
    main()