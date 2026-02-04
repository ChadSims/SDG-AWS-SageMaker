import argparse
import os 
import pandas as pd
from sdv.metadata import Metadata

from lib.utils import dump_config
# from lib.info import DATASETS_PATH

DATASETS_PATH = os.getenv("DATASETS_PATH", "/opt/ml/input/data/datasets")


def main():
    parser = argparse.ArgumentParser(description="Generate metadata from a DataFrame.")
    parser.add_argument('--dataset', type=str, required=True, help='The name of the dataset (without extension).')
    parser.add_argument('--target_column', type=str, required=True, help='Name of the target column.')

    args = parser.parse_args()

    dataset = args.dataset
    target_column = args.target_column

    dataset_path = os.path.join(DATASETS_PATH, f"{dataset}/{dataset}.csv")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}. Please provide a valid dataset path.")
    
    df = pd.read_csv(dataset_path)

    N, d = df.shape

    percent_rows_with_missing = df.isnull().any(axis=1).mean() * 100

    sdv_metadata = Metadata.detect_from_dataframe(df)
    sdtypes = {col: sdv_metadata.to_dict()['tables']['table']['columns'][col]['sdtype'] for col in df.columns}

    num_features = [col for col, sdtype in sdtypes.items() if sdtype == 'numerical' and col != target_column]
    cat_features = [col for col, sdtype in sdtypes.items() if sdtype == 'categorical' and col != target_column]

    target_type = sdtypes[target_column]

    task = 'regression' if target_type == 'numerical' else 'classification'

    metadata = {
        'num_features': num_features,
        'cat_features': cat_features,
        'target_column': target_column,
        'task': task,
        'N': N,
        'd': d,
        'missing_percentage': percent_rows_with_missing,
    }

    metadata_path = os.path.join(DATASETS_PATH, f"{dataset}/metadata.toml")

    dump_config(metadata, metadata_path)


if __name__ == "__main__":
    main()