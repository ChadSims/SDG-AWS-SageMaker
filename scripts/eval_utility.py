import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

from lib.evaluators import train_ridge, train_svr, train_rfr, train_mlpr, train_logistic, train_svc, train_rfc, train_mlpc
from lib.utils import load_config, dump_config, dump_json, load_dataset
from lib.preprocess import clean, split, transform

DATASETS_PATH = os.getenv("DATASETS_PATH", "/opt/ml/input/data/datasets")
PARAMS_PATH = os.getenv("PARAMS_PATH", "/opt/ml/output/data/params")
EXP_PATH = os.getenv("EXP_PATH", "/opt/ml/output/data/exp")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - [Proc:%(process)d] - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

logger = logging.getLogger(__name__)


def regressor_utility(data: dict, dataset: str,  params_path: str, random_state=42):
    
    ridge_path = os.path.join(params_path, f"ridge/{dataset}.toml")

    if not os.path.exists(ridge_path):
        raise FileNotFoundError(f"Ridge parameters not found at {ridge_path}. Please tune the ridge regressor first.")
    
    ridge_params = load_config(ridge_path)
    ridge_params['random_state'] = random_state
    ridge_results = train_ridge(ridge_params, **data)

    svr_path = os.path.join(params_path, f"svr/{dataset}.toml")
    svr_params = load_config(svr_path)
    svr_results = train_svr(svr_params, **data)

    rfr_path = os.path.join(params_path, f"rfr/{dataset}.toml")
    rfr_params = load_config(rfr_path)
    rfr_params['random_state'] = random_state
    rfr_results = train_rfr(rfr_params, **data)

    mlpr_path = os.path.join(params_path, f"mlpr/{dataset}.toml")
    mlpr_params = load_config(mlpr_path)
    mlpr_params['random_state'] = random_state
    mlpr_results = train_mlpr(mlpr_params, **data)

    regressor_results = {
        'ridge': ridge_results,
        'svr': svr_results,
        'rfr': rfr_results,
        'mlpr': mlpr_results,
    }

    return regressor_results

def classifier_utility(data: dict, dataset: str,  params_path: str, random_state=42):

    logistic_path = os.path.join(params_path, f"logistic/{dataset}.toml")

    if not os.path.exists(logistic_path):
        raise FileNotFoundError(f"Logistic regression parameters not found at {logistic_path}. Please tune the logistic regression model first.")
    
    logistic_params = load_config(logistic_path)
    logistic_params['random_state'] = random_state
    logistic_results = train_logistic(logistic_params, **data)

    svc_path = os.path.join(params_path, f"svc/{dataset}.toml")
    svc_params = load_config(svc_path)
    svc_params['random_state'] = random_state
    svc_results = train_svc(svc_params, **data)

    rfc_path = os.path.join(params_path, f"rfc/{dataset}.toml")
    rfc_params = load_config(rfc_path)
    rfc_params['random_state'] = random_state
    rfc_results = train_rfc(rfc_params, **data)

    mlpc_path = os.path.join(params_path, f'mlpc/{dataset}.toml')
    mlpc_params = load_config(mlpc_path)
    mlpc_params['random_state'] = random_state
    mlpc_results = train_mlpc(mlpc_params, **data)

    classifier_results = {
        'logistic': logistic_results,
        'svc': svc_results,
        'rfc': rfc_results,
        'mlpc': mlpc_results,
    }

    return classifier_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='The name of the dataset (without extension).')
    parser.add_argument('--model', type=str, help='Name of the model (synthesiser) used to generate the synthetic data. Only required when tuning evaluators on a synthetic dataset.')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility.')

    args = parser.parse_args()

    dataset = args.dataset
    
    metadata_path = os.path.join(DATASETS_PATH, f"{dataset}/metadata.toml")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}. Please generate metadata first.")

    metadata = load_config(metadata_path)
    target_column = metadata['target_column']
    task = metadata['task']
    num_features = metadata['num_features']
    cat_features = metadata['cat_features']

    train, val, test = load_dataset(dataset)

    X_train = train.drop(columns=[target_column])
    y_train = train[target_column]
    X_val = val.drop(columns=[target_column])
    y_val = val[target_column]
    X_test = test.drop(columns=[target_column])
    y_test = test[target_column]


    if args.model:
        # If a model is specified, we assume this is a synthetic dataset
        model = args.model
        synthetic_dataset_path = os.path.join(EXP_PATH, f'{dataset}/{model}/sample.csv')

        if not os.path.exists(synthetic_dataset_path):
            logger.info(f"Synthetic dataset not found at {synthetic_dataset_path}.")
            
            # Attempt to find the synthetic dataset in the input directory
            synthetic_dataset_path = synthetic_dataset_path.replace('output', 'input')

            logger.info(f"Attempting to load synthetic dataset from {synthetic_dataset_path}")

            if not os.path.exists(synthetic_dataset_path):
                raise FileNotFoundError(f"Synthetic dataset file not found at {synthetic_dataset_path}. Please generate the synthetic dataset first or move it to input/data/exp/dataset/model/sample.csv.")
        
        df = pd.read_csv(synthetic_dataset_path)

        clean(df)

        S_X_train = df.drop(columns=[target_column])
        S_y_train = df[target_column]

        # Transform the synthetic data, transformers fit on synthetic data, real test data is only transformed. val data is used as placeholder but discarded
        X_train, _, X_test, y_train, _, y_test = transform(S_X_train, X_val, X_test, S_y_train, y_val, y_test, num_features, cat_features, task)
            
        data = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }   

        params_path = os.path.join(EXP_PATH, f'{dataset}/{model}/evaluators')

        if not os.path.exists(params_path):
            logger.info(f"Parameters for evaluators not found at {params_path}.")
        
            # Attempt to find the parameters in the input directory
            params_path = params_path.replace('output', 'input')

            logger.info(f"Attempting to load parameters from {params_path}")

            if not os.path.exists(params_path):
                raise FileNotFoundError(f"Parameters file not found at {params_path}. Please generate parameters first or move them to input/data/exp/dataset/model/evaluators.")

        results_path = os.path.join(EXP_PATH, f'{dataset}/{model}/utility_result.json')
        os.makedirs(os.path.dirname(results_path), exist_ok=True)

    else:
        X_train, X_val, X_test, y_train, y_val, y_test = transform(X_train, X_val, X_test, y_train, y_val, y_test, num_features, cat_features, task)

        if isinstance(X_train, pd.DataFrame):
            X_train = pd.concat([X_train, X_val], axis=0, ignore_index=True)
            y_train = pd.concat([y_train, y_val], axis=0, ignore_index=True)
        elif isinstance(X_train, np.ndarray):
            X_train = np.concatenate((X_train, X_val), axis=0)
            y_train = np.concatenate((y_train, y_val), axis=0)

        data = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }  

        params_path = os.path.join(PARAMS_PATH, 'evaluators')

        if not os.path.exists(params_path):
            logger.info(f"Parameters for evaluators not found at {params_path}.")
        
            # Attempt to find the parameters in the input directory
            params_path = params_path.replace('output', 'input')

            logger.info(f"Attempting to load parameters from {params_path}")

            if not os.path.exists(params_path):
                raise FileNotFoundError(f"Parameters file not found at {params_path}. Please generate parameters first or move them to input/data/params/evaluators.")

        results_path = os.path.join(EXP_PATH, f'{dataset}/utility_result.json')
        os.makedirs(os.path.dirname(results_path), exist_ok=True)


    if task == 'regression':
        utility_results = regressor_utility(data, dataset, params_path, random_state=args.random_state)
    elif task == 'classification':
        utility_results = classifier_utility(data, dataset, params_path, random_state=args.random_state)


    dump_json(utility_results, results_path)


if __name__ == "__main__":
    main()