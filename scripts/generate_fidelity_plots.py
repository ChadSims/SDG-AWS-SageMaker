import argparse
from importlib import import_module
import os 
import pandas as pd

from lib.preprocess import clean
from lib.plotting import plot_feature_distribution_comparison_multiple
from lib.utils import load_config, load_dataset
# from lib.info import DATASETS_PATH, EXP_PATH

DATASETS_PATH = os.getenv("DATASETS_PATH", "/opt/ml/input/data/datasets")
EXP_PATH = os.getenv("EXP_PATH", "/opt/ml/output/data/exp")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='The name of the dataset (without extension).')
    parser.add_argument('--models', type=str, required=True, help='A comma-separated list of models.')

    args = parser.parse_args()

    dataset = args.dataset
    models = args.models.split(',')

    dataset_path = os.path.join(DATASETS_PATH, f"{dataset}/{dataset}.csv")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}. Please provide a valid dataset path.")

    real_data = pd.read_csv(dataset_path)
    clean(real_data)

    synthetic_datasets = []
    synthetic_dataset_paths = [os.path.join(EXP_PATH, f'{dataset}/{model}/sample.csv') for model in models]

    conditions = [os.path.exists(path) for path in synthetic_dataset_paths]
    if not all(conditions):
        raise FileNotFoundError(f"One or more synthetic datasets not found. Please generate synthetic data for all models first.")
    
    for path in synthetic_dataset_paths:
        synthetic_data = pd.read_csv(path)
        clean(synthetic_data)
        synthetic_datasets.append(synthetic_data)

    conditions = [real_data.shape[1] == synthetic_data.shape[1] for synthetic_data in synthetic_datasets]
    if not all(conditions):
        raise ValueError("Real and synthetic datasets must have the same number of columns.")
    
    conditions = [real_data.columns.tolist() == synthetic_data.columns.tolist() for synthetic_data in synthetic_datasets]
    if not all(conditions):
        raise ValueError("Real and synthetic datasets must have the same column names.")
    
    plot_feature_distribution_comparison_multiple(synthetic_datasets, models)





if __name__ == "__main__":
    main()