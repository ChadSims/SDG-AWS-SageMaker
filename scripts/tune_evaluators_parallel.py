import argparse 
import logging
import os
import sys

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

import wandb

from lib.evaluators import train_ridge, train_logistic, train_svr, train_svc, train_rfr, train_rfc, train_mlpr, train_mlpc
from lib.plotting import optuna_timeline_plot
from lib.preprocess import clean, split, transform, preprocess
from lib.utils import load_config, load_dataset, dump_config

DATASETS_PATH = os.getenv("DATASETS_PATH", "/opt/ml/input/data/datasets")
PARAMS_PATH = os.getenv("PARAMS_PATH", "/opt/ml/output/data/params")
EXP_PATH = os.getenv("EXP_PATH", "/opt/ml/output/data/exp")

JOURNAL_STORAGE = os.getenv("JOURNAL_STORAGE", "/opt/ml/output/data/journal.log")
N_TRIALS_PER_WORKER = int(os.getenv("N_TRIALS_PER_WORKER", 10))
N_WORKERS = int(os.getenv("N_WORKERS", 5))
RUN_LOCAL = os.getenv("RUN_LOCAL", "false").lower() == "true"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - [Proc:%(process)d] - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

logger = logging.getLogger(__name__)

wandb.login()

def ridge_objective(trial, X_train, y_train, X_val, y_val, random_state):
    logger.info(f"Ridge Trial {trial.number} (started by Proc:{os.getpid()})")
    alpha = trial.suggest_float("alpha", 0, 10)
    fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])
    params = {"alpha": alpha, "fit_intercept": fit_intercept, "random_state": random_state}
    res = train_ridge(params, X_train, y_train, X_val, y_val)
    return res['mse']

def logistic_objective(trial, X_train, y_train, X_val, y_val, random_state):
    logger.info(f"Logistic Regression Trial {trial.number} (started by Proc:{os.getpid()})")
    max_iter = trial.suggest_int("max_iter", 100, 1000)
    C = trial.suggest_float("C", 1e-5, 1e-1, log=True)
    tol = trial.suggest_float("tol", 1e-5, 1e-1, log=True)
    params = {"max_iter": max_iter, "C": C, "tol": tol, "random_state": random_state}
    res = train_logistic(params, X_train, y_train, X_val, y_val)
    return res['f1']

def svr_objective(trial, X_train, y_train, X_val, y_val, random_state):
    logger.info(f"SVR Trial {trial.number} (started by Proc:{os.getpid()})")
    C = trial.suggest_float("C", 1e-5, 1e-1, log=True)
    epsilon = trial.suggest_float("epsilon", 1e-5, 1e-1, log=True)
    params = {"C": C, "epsilon": epsilon}
    res = train_svr(params, X_train, y_train, X_val, y_val)
    return res['mse']

def svc_objective(trial, X_train, y_train, X_val, y_val, random_state):
    logger.info(f"SVC Trial {trial.number} (started by Proc:{os.getpid()})")
    C = trial.suggest_float("C", 1e-5, 1e-1, log=True)
    kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
    gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
    params = {"C": C, "kernel": kernel, "gamma": gamma, "random_state": random_state}
    res = train_svc(params, X_train, y_train, X_val, y_val)
    return res['f1']

def rf_objective(trial, task, X_train, y_train, X_val, y_val, random_state):
    logger.info(f"RF Trial {trial.number} (started by Proc:{os.getpid()})")
    n_estimators = trial.suggest_int("n_estimators", 10, 200)
    max_depth = trial.suggest_int("max_depth", 4, 64)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 8)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 8)
    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "random_state": random_state
    }
    score = None
    if task == 'classification':
        res = train_rfc(params, X_train, y_train, X_val, y_val)
        score = res['f1']
    else:
        res = train_rfr(params, X_train, y_train, X_val, y_val)
        score = res['mse']
    return score

def mlp_objective(trial, task, X_train, y_train, X_val, y_val, random_state):
    logger.info(f"MLP Trial {trial.number} (started by Proc:{os.getpid()})")
    max_iter = trial.suggest_int("max_iter", 50, 200)
    alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
    params = {"alpha": alpha, "max_iter": max_iter, "random_state": random_state}
    score = None
    if task == 'classification':
        res = train_mlpc(params, X_train, y_train, X_val, y_val)
        score = res['f1']
    else:
        res = train_mlpr(params, X_train, y_train, X_val, y_val)
        score = res['mse']
    return score

def optuna_worker(
    worker_id,
    model_type: str,
    X_train_data, y_train_data, X_val_data, y_val_data,
    project_name: str,
    random_state,
    storage, # The Optuna storage object
    n_trials_per_worker # How many trials this worker will run
):
    logger.info(f"Worker {worker_id} for {model_type} optimization (Proc:{os.getpid()}) starting.")

    if model_type == 'ridge':
        objective_func = lambda trial: ridge_objective(trial, X_train_data, y_train_data, X_val_data, y_val_data, random_state)
        metric_name = 'mse'
        direction = 'minimize'
    elif model_type == 'svr':
        objective_func = lambda trial: svr_objective(trial, X_train_data, y_train_data, X_val_data, y_val_data, random_state)
        metric_name = 'mse'
        direction = 'minimize'
    elif model_type == 'rfr':
        objective_func = lambda trial: rf_objective(trial, 'regression', X_train_data, y_train_data, X_val_data, y_val_data, random_state)
        metric_name = 'mse'
        direction = 'minimize'
    elif model_type == 'mlpr':
        objective_func = lambda trial: mlp_objective(trial, 'regression', X_train_data, y_train_data, X_val_data, y_val_data, random_state)
        metric_name = 'mse'
        direction = 'minimize'
    elif model_type == 'logistic':
        objective_func = lambda trial: logistic_objective(trial, X_train_data, y_train_data, X_val_data, y_val_data, random_state)
        metric_name = 'f1'
        direction = 'maximize'
    elif model_type == 'svc':
        objective_func = lambda trial: svc_objective(trial, X_train_data, y_train_data, X_val_data, y_val_data, random_state)
        metric_name = 'f1'
        direction = 'maximize'
    elif model_type == 'rfc':
        objective_func = lambda trial: rf_objective(trial, 'classification', X_train_data, y_train_data, X_val_data, y_val_data, random_state)
        metric_name = 'f1'
        direction = 'maximize'
    elif model_type == 'mlpc':
        objective_func = lambda trial: mlp_objective(trial, 'classification', X_train_data, y_train_data, X_val_data, y_val_data, random_state)
        metric_name = 'f1'
        direction = 'maximize'
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    wandb_kwargs = {
        "project": project_name,
    }
    wandbc = WeightsAndBiasesCallback(
        metric_name=metric_name,
        wandb_kwargs=wandb_kwargs,
        as_multirun=True
    )

    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=worker_id), 
        storage=JournalStorage(JournalFileBackend(file_path=storage)),
        study_name=project_name,
        load_if_exists=True # This is ESSENTIAL for parallelization
    )
    study.optimize(
        objective_func,
        n_trials=n_trials_per_worker,
        show_progress_bar=False,
        callbacks=[wandbc]
    )

    logger.info(f"Worker {worker_id} / process: {os.getpid()} completed {n_trials_per_worker} trials for {model_type}.")

def tune_model_parallel(model_type: str, X_train, y_train, X_val, y_val, study_name, task, random_state=42):

    logger.info(f"\n--- Starting parallel tuning for {model_type} model ---")

    project_name = f"local-sdg-evaluators-{study_name}" if RUN_LOCAL else f"sagemaker-sdg-evaluators-{study_name}"

    try:
        optuna.delete_study(study_name=project_name, storage=JournalStorage(JournalFileBackend(file_path=JOURNAL_STORAGE)))
    except:
        pass    

    worker_args = []
    for i in range(N_WORKERS):
        worker_args.append((
            i, # worker_id
            model_type,
            X_train, y_train, X_val, y_val,
            project_name,
            random_state,
            JOURNAL_STORAGE,
            N_TRIALS_PER_WORKER
        ))

    logger.info(f"Creating process pool with {N_WORKERS} workers")
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        executor.map(optuna_worker,  *zip(*worker_args))

    study = optuna.load_study(
        study_name=project_name,
        storage=JournalStorage(JournalFileBackend(file_path=JOURNAL_STORAGE))
    )

    # try:                 
    #     objective_value = 'MSE' if task == 'regression' else 'F1 Score'

    #     optim_history = optuna.visualization.plot_optimization_history(study, target_name=objective_value)
    #     optim_history.update_layout(title_text=None)
    #     param_importance = optuna.visualization.plot_param_importances(study)
    #     param_importance.update_layout(title_text=None)

    #     trials_df = study.trials_dataframe()
    #     timeline_plot = optuna_timeline_plot(trials_df)

    #     run = wandb.init(
    #         project=project_name,
    #         name='summary_plots',
    #         tags=['summary_plots'],
    #     )
    #     run.log({
    #         "Optimisation History": optim_history,
    #         "Parameter Importance": param_importance,
    #         "Timeline": timeline_plot,
    #     })
    #     run.finish()
    # except Exception as e:
    #     logger.error(f"Error during plotting: {e}")
    #     logger.info("Skipping plots due to error.")

    best_params = study.best_params
    best_score = study.best_value

    return best_params, best_score

def tune_regressors(data: dict, dataset: str, params_path: str, model, random_state: int = 42):
    """
    data = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
    }
    dataset: name of the dataset
    params_path: path to save the parameters
    model: name of the model (synthesiser) used to generate the synthetic data, if applicable
    """
    # tune ridge
    study_name = f"ridge-{dataset}" if not model else f"ridge-{dataset}-{model}"
    best_params, best_score = tune_model_parallel('ridge', **data, study_name=study_name, task='regression', random_state=random_state)
    ridge_path = os.path.join(params_path, f"ridge/{dataset}.toml")
    dump_config(best_params, ridge_path)

    # tune svr
    study_name = f"svr-{dataset}" if not model else f"svr-{dataset}-{model}"
    best_params, best_score = tune_model_parallel('svr', **data, study_name=study_name, task='regression', random_state=random_state)
    svr_path = os.path.join(params_path, f"svr/{dataset}.toml")
    dump_config(best_params, svr_path)

    # tune rfr
    study_name = f"rfr-{dataset}" if not model else f"rfr-{dataset}-{model}"
    best_params, best_score = tune_model_parallel('rfr', **data, study_name=study_name, task='regression', random_state=random_state)
    rfr_path = os.path.join(params_path, f"rfr/{dataset}.toml")
    dump_config(best_params, rfr_path)

    # tune mlpr
    study_name = f"mlpr-{dataset}" if not model else f"mlpr-{dataset}-{model}"
    best_params, best_score = tune_model_parallel('mlpr', **data, study_name=study_name, task='regression', random_state=random_state)
    mlpr_path = os.path.join(params_path, f"mlpr/{dataset}.toml")
    dump_config(best_params, mlpr_path)

def tune_classifiers(data: dict, dataset: str, params_path: str, model, random_state: int = 42):
    # logistic
    study_name = f"logistic-{dataset}" if not model else f"logistic-{dataset}-{model}"
    best_params, best_score = tune_model_parallel('logistic', **data, study_name=study_name, task='classification', random_state=random_state)
    logistic_path = os.path.join(params_path, f"logistic/{dataset}.toml")
    dump_config(best_params, logistic_path)

    # svc
    study_name = f"svc-{dataset}" if not model else f"svc-{dataset}-{model}"
    best_params, best_score = tune_model_parallel('svc', **data, study_name=study_name, task='classification', random_state=random_state)
    svc_path = os.path.join(params_path, f"svc/{dataset}.toml")
    dump_config(best_params, svc_path)

    # rfc
    study_name = f"rfc-{dataset}" if not model else f"rfc-{dataset}-{model}"
    best_params, best_score = tune_model_parallel('rfc', **data, study_name=study_name, task='classification', random_state=random_state)
    rfc_path = os.path.join(params_path, f"rfc/{dataset}.toml")
    dump_config(best_params, rfc_path)

    # mlpc
    study_name = f"mlpc-{dataset}" if not model else f"mlpc-{dataset}-{model}"
    best_params, best_score = tune_model_parallel('mlpc', **data, study_name=study_name, task='classification', random_state=random_state)
    mlpc_path = os.path.join(params_path, f"mlpc/{dataset}.toml")
    dump_config(best_params, mlpc_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='The name of the dataset (without extension).')
    parser.add_argument('--model', type=str, default=None, help='Name of the model (synthesiser) used to generate the synthetic data. Only required when tuning evaluators on a synthetic dataset.')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility.')

    args = parser.parse_args()

    dataset = args.dataset
    model = args.model
    random_state = args.random_state

    metadata_path = os.path.join(DATASETS_PATH, f"{dataset}/metadata.toml")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}. Please generate metadata first.")

    metadata = load_config(metadata_path)
    task = metadata['task']
    target_column = metadata['target_column']
    num_features = metadata['num_features']
    cat_features = metadata['cat_features']

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
        X_train, X_val, X_test, y_train, y_val, y_test = preprocess(df, metadata=metadata)
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
        logger.info(f"Calling tune regressors for dataset: {dataset}, model: {model}")
        tune_regressors(data, dataset, params_path, model, random_state=random_state)
    elif task == 'classification':
        tune_classifiers(data, dataset, params_path, model, random_state=random_state)