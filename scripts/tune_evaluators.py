import argparse 
import logging
import os
import sys

import numpy as np
import pandas as pd

import wandb

from lib.evaluators import tune_ridge, tune_svr, tune_rfr, tune_mlpr, tune_logistic, tune_svc, tune_rfc, tune_mlpc
from lib.preprocess import clean, split, transform, preprocess
from lib.utils import load_config, load_dataset, dump_config

DATASETS_PATH = os.getenv("DATASETS_PATH", "/opt/ml/input/data/datasets")
PARAMS_PATH = os.getenv("PARAMS_PATH", "/opt/ml/output/data/params")
EXP_PATH = os.getenv("EXP_PATH", "/opt/ml/output/data/exp")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - [Proc:%(process)d] - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

logger = logging.getLogger(__name__)

wandb.login()

def tune_regressors(data: dict, dataset: str, params_path: str, model, random_state: int = 42):

    study_name = f"ridge-{dataset}" if not model else f"ridge-{dataset}-{model}" 
    best_params, _ = tune_ridge(**data, study_name=study_name, random_state=random_state)
    ridge_path = os.path.join(params_path, f"ridge/{dataset}.toml")
    dump_config(best_params, ridge_path)

    study_name = f"svr-{dataset}" if not model else f"svr-{dataset}-{model}"
    best_params, _ = tune_svr(**data, study_name=study_name)
    svr_path = os.path.join(params_path, f"svr/{dataset}.toml")
    dump_config(best_params, svr_path)

    study_name = f"rfr-{dataset}" if not model else f"rfr-{dataset}-{model}"
    best_params, _ = tune_rfr(**data, study_name=study_name, random_state=random_state)
    rfr_path = os.path.join(params_path, f"rfr/{dataset}.toml")
    dump_config(best_params, rfr_path)

    study_name = f"mlpr-{dataset}" if not model else f"mlpr-{dataset}-{model}"
    best_params, _ = tune_mlpr(**data, study_name=study_name)
    mlpr_path = os.path.join(params_path, f"mlpr/{dataset}.toml")
    dump_config(best_params, mlpr_path)

def tune_classifiers(data: dict, dataset: str, params_path: str, model, random_state: int = 42):

    study_name = f"logistic-{dataset}" if not model else f"logistic-{dataset}-{model}"
    best_params, _ = tune_logistic(**data, study_name=study_name, random_state=random_state)
    logistic_path = os.path.join(params_path, f"logistic/{dataset}.toml")
    dump_config(best_params, logistic_path)

    study_name = f"svc-{dataset}" if not model else f"svc-{dataset}-{model}"
    best_params, _ = tune_svc(**data, study_name=study_name, random_state=random_state)
    svc_path = os.path.join(params_path, f"svc/{dataset}.toml")
    dump_config(best_params, svc_path)

    study_name = f"rfc-{dataset}" if not model else f"rfc-{dataset}-{model}"
    best_params, _ = tune_rfc(**data, study_name=study_name, random_state=random_state)
    rfc_path = os.path.join(params_path, f"rfc/{dataset}.toml")
    dump_config(best_params, rfc_path)

    study_name = f"mlpc-{dataset}" if not model else f"mlpc-{dataset}-{model}"
    best_params, _ = tune_mlpc(**data, study_name=study_name, random_state=random_state)
    mlpc_path = os.path.join(params_path, f"mlpc/{dataset}.toml")
    dump_config(best_params, mlpc_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='The name of the dataset (without extension).')
    parser.add_argument('--model', type=str, default=None, help='Name of the model (synthesiser) used to generate the synthetic data. Only required when tuning evaluators on a synthetic dataset.')

    args = parser.parse_args()

    dataset = args.dataset
    model = args.model

    metadata_path = os.path.join(DATASETS_PATH, f"{dataset}/metadata.toml")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}. Please generate metadata first.")

    metadata = load_config(metadata_path)
    task = metadata['task']
    target_column = metadata['target_column']
    num_features = metadata['num_features']
    cat_features = metadata['cat_features']
    random_state = metadata['random_state']
    test_size = metadata['test_size']

    if model:
        # If a model is specified, we assume this is a synthetic dataset

        synthetic_dataset_path = os.path.join(EXP_PATH, f'{dataset}/{model}/sample.csv')

        if not os.path.exists(synthetic_dataset_path):
            logger.info(f"Synthetic dataset not found at {synthetic_dataset_path}.")
            
            # Attempt to find the synthetic dataset in the input directory
            synthetic_dataset_path = synthetic_dataset_path.replace('output', 'input')

            logger.info(f"Attempting to load synthetic dataset from {synthetic_dataset_path}")

            if not os.path.exists(synthetic_dataset_path):
                raise FileNotFoundError(f"Synthetic dataset file not found at {synthetic_dataset_path}. Please generate the synthetic dataset first or move it to input/data/exp/dataset/model/sample.csv.")
        
        df = pd.read_csv(synthetic_dataset_path)
        X_train, X_val, X_test, y_train, y_val, y_test = preprocess(df, metadata=metadata, test_size=test_size, random_state=random_state)
        # synthetic data can be concat before transform - currently val not taken into account when transformed
        if isinstance(X_train, pd.DataFrame):
            X_train = pd.concat([X_train, X_val], axis=0, ignore_index=True)
            y_train = pd.concat([y_train, y_val], axis=0, ignore_index=True)
        elif isinstance(X_train, np.ndarray):
            X_train = np.concatenate((X_train, X_val), axis=0)
            y_train = np.concatenate((y_train, y_val), axis=0)
           
        data = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_test,
            "y_val": y_test,
        }   

        params_path = os.path.join(EXP_PATH, f'{dataset}/{model}/evaluators')
        os.makedirs(params_path, exist_ok=True)   

    else:
        # If no model is specified, we assume this is the real dataset
        train, val, test = load_dataset(dataset)

        X_train = train.drop(columns=[target_column])
        y_train = train[target_column]
        X_val = val.drop(columns=[target_column])
        y_val = val[target_column]
        X_test = test.drop(columns=[target_column])
        y_test = test[target_column]

        X_train, X_val, _, y_train, y_val, _ = transform(X_train, X_val, X_test, y_train, y_val, y_test, num_features, cat_features, task)

        data = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
        }

        params_path = os.path.join(PARAMS_PATH, 'evaluators')
        os.makedirs(params_path, exist_ok=True)


    if task == 'regression':
        tune_regressors(data, dataset, params_path, model, random_state=random_state)
    elif task == 'classification':
        tune_classifiers(data, dataset, params_path, model, random_state=random_state)


if __name__ == "__main__":
    main()