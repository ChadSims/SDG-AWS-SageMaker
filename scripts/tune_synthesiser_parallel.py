import argparse
import logging
import os 
import sys

from concurrent.futures import ProcessPoolExecutor

import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

import pandas as pd

import wandb

from lib.evaluation import cal_fidelity
# from lib.plotting import optuna_timeline_plot
from lib.preprocess import clean
from lib.utils import load_config, dump_config, load_dataset
from synthesisers.ctgan import train as ctgan_train
from synthesisers.tvae import train as tvae_train
from synthesisers.gaussian_copula import train as gaussian_copula_train
from synthesisers.copula_gan import train as copula_gan_train
from synthesisers.binary_diffusion import train as binary_diffusion_train

DATASETS_PATH = os.getenv("DATASETS_PATH", "/opt/ml/input/data/datasets")
PARAMS_PATH = os.getenv("PARAMS_PATH", "/opt/ml/output/data/params")
EXP_PATH = os.getenv("EXP_PATH", "/opt/ml/output/data/exp")

JOURNAL_STORAGE = os.getenv("JOURNAL_STORAGE", "/opt/ml/output/data/journal.log")
N_TRIALS_PER_WORKER = int(os.getenv("N_TRIALS_PER_WORKER", 25))
N_WORKERS = int(os.getenv("N_WORKERS", 2))
N_DATALOADERS = int(os.getenv("N_DATALOADERS", 2))
RUN_LOCAL = os.getenv("RUN_LOCAL", "false").lower() == "true"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - [Proc:%(process)d] - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

logger = logging.getLogger(__name__)

wandb.login()

def ctgan_objective(trial, data: pd.DataFrame):

    model_params = {}
    model_params["generator_lr"] = trial.suggest_float("generator_lr", 1e-5, 1e-3, log=True)
    model_params["discriminator_lr"] = trial.suggest_float("discriminator_lr", 1e-5, 1e-3, log=True)
    model_params["generator_decay"] = 1e-6
    model_params["discriminator_decay"] = trial.suggest_categorical("discriminator_decay", [0, 1e-6])
    # model_params["batch_size"] = trial.suggest_categorical("batch_size", [40, 80, 160, 320, 500])
    # model_params["epochs"] = trial.suggest_int("epochs", 100, 500)

    model_params["batch_size"] = 500
    model_params["epochs"] = 250  # Fixed for tuning, can be adjusted later
    model_params["embedding_dim"] = 128
    model_params["generator_dim"] = [128, 128]
    model_params["discriminator_dim"] = [256, 256]

    trial.set_user_attr("best_params", model_params)

    synthetic_data, loss = ctgan_train(data, model_params, None, tune=True)

    loss["trial"] = trial.number

    loss_table = wandb.Table(dataframe=loss)
    wandb.log(
        {
            f"ctgan-loss-tune": loss_table,
        }
    )

    fidelity_score = cal_fidelity(data, synthetic_data, tune=True)

    wandb.log({'wasserstein': fidelity_score})

    return fidelity_score


def tvae_objective(trial, data: pd.DataFrame):
    model_params = {}

    model_params["l2scale"] = trial.suggest_float("l2scale", 1e-6, 1e-3, log=True)
    # model_params["batch_size"] = trial.suggest_categorical("batch_size", [40, 80, 160, 320, 500])
    # model_params["epochs"] = trial.suggest_int("epochs", 100, 500)
    model_params["loss_factor"]= trial.suggest_float("loss_factor", 1.0, 5.0)

    model_params["batch_size"] = 500
    model_params["epochs"] = 250  # Fixed for tuning, can be adjusted later
    model_params["embedding_dim"] = 128
    model_params["compress_dims"] = [128, 128]
    model_params["decompress_dims"] = [128, 128]

    trial.set_user_attr("best_params", model_params)

    synthetic_data, loss = tvae_train(data, model_params, None, tune=True)

    multi_agg_loss_per_epoch = loss.groupby('Epoch')['Loss'].agg(['mean', 'min', 'max', 'std']).reset_index()
    multi_agg_loss_per_epoch["trial"] = trial.number

    loss_table = wandb.Table(dataframe=multi_agg_loss_per_epoch)
    wandb.log(
        {
            f"tvae-loss-tune": loss_table,
        }
    )

    fidelity_score = cal_fidelity(data, synthetic_data, tune=True)

    wandb.log({'wasserstein': fidelity_score})

    return fidelity_score


def gaussian_copula_objective(trial, data: pd.DataFrame, num_features: list):
        model_params = {}
        model_params["numerical_distributions"] = {}

        for col in num_features:
            model_params["numerical_distributions"][col] = trial.suggest_categorical(
                f"numerical_distribution_{col}",
                ["norm", "truncnorm", "uniform", "gamma"] # "beta" and "gaussian_kde" removed: "beta" can fail sometimes, "gaussian_kde" slows optimisation significantly
            )

        trial.set_user_attr("best_params", model_params)

        synthetic_data = gaussian_copula_train(data, model_params, None, tune=True)

        fidelity_score = cal_fidelity(data, synthetic_data, tune=True)

        wandb.log({'wasserstein': fidelity_score})

        return fidelity_score


def copula_gan_objective(trial, data: pd.DataFrame, num_features: list):
    model_params = {}
    model_params["generator_lr"] = trial.suggest_float("generator_lr", 1e-5, 1e-3, log=True)
    model_params["discriminator_lr"] = trial.suggest_float("discriminator_lr", 1e-5, 1e-3, log=True)
    model_params["generator_decay"] = 1e-6
    model_params["discriminator_decay"] = trial.suggest_categorical("discriminator_decay", [0, 1e-6])
    # model_params["batch_size"] = trial.suggest_categorical("batch_size", [40, 80, 160, 320, 500])
    # model_params["epochs"] = trial.suggest_int("epochs", 100, 500)

    model_params["batch_size"] = 500
    model_params["epochs"] = 250  # Fixed for tuning, can be adjusted later
    model_params["embedding_dim"] = 128
    model_params["generator_dim"] = [128, 128]
    model_params["discriminator_dim"] = [256, 256]

    model_params["numerical_distributions"] = {}

    for col in num_features:
        model_params["numerical_distributions"][col] = trial.suggest_categorical(
            f"numerical_distribution_{col}",
            ["norm", "truncnorm", "uniform", "gamma"] # "beta" and "gaussian_kde" removed: "beta" can fail sometimes, "gaussian_kde" slows optimisation significantly
        )

    trial.set_user_attr("best_params", model_params)

    synthetic_data, loss = copula_gan_train(data, model_params, None, tune=True)

    loss["trial"] = trial.number

    loss_table = wandb.Table(dataframe=loss)
    wandb.log(
        {
            f"copula_gan-loss-tune": loss_table,
        }
    )

    fidelity_score = cal_fidelity(data, synthetic_data, tune=True)

    wandb.log({'wasserstein': fidelity_score})

    return fidelity_score


def binary_diffusion_objective(trial, data: pd.DataFrame, metadata: dict, exp_path: str, worker_id: int):
    model_params = {}
    model_params["lr"] = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    model_params["batch_size"] = trial.suggest_categorical("batch_size", [64, 128, 256, 512])

    trial.set_user_attr("best_params", model_params)

    synthetic_data = binary_diffusion_train(data, model_params, metadata, exp_path, tune=True, worker_id=worker_id)

    clean(synthetic_data)

    fidelity_score = cal_fidelity(data, synthetic_data, tune=True)

    wandb.log({'wasserstein': fidelity_score})

    return fidelity_score


def optuna_worker(
    worker_id: int,
    model_type: str,
    data: pd.DataFrame,
    metadata: dict,
    project_name: str,
    storage: str,
    n_trials_per_worker,
    exp_path: str
):
    logger.info(f"Worker {worker_id} started for model {model_type} with study {study_name}")

    metric_name = "Wasserstein"
    direction = 'minimize'

    if model_type == 'ctgan':
        objective_func = lambda trial: ctgan_objective(trial, data)
    elif model_type == 'tvae':
        objective_func = lambda trial: tvae_objective(trial, data)
    elif model_type == 'gaussian_copula':
        num_features = metadata['num_features']
        num_features.append(metadata['target_column']) if metadata['task'] == 'regression' else None
        objective_func = lambda trial: gaussian_copula_objective(trial, data, num_features)
    elif model_type == 'copula_gan':
        num_features = metadata['num_features']
        num_features.append(metadata['target_column']) if metadata['task'] == 'regression' else None
        objective_func = lambda trial: copula_gan_objective(trial, data, num_features)
    elif model_type == 'binary_diffusion':
        objective_func = lambda trial: binary_diffusion_objective(trial, data, metadata, exp_path, worker_id)


    wandb_kwargs = {
        "project": project_name,
    }
    wandbc = WeightsAndBiasesCallback(
        metric_name=metric_name,
        wandb_kwargs=wandb_kwargs,
        as_multirun=True
    )

    decorated_objective = wandbc.track_in_wandb()(objective_func) # allows wandb.log inside objective function

    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=worker_id), 
        storage=JournalStorage(JournalFileBackend(file_path=storage)),
        study_name=project_name,
        load_if_exists=True # This is ESSENTIAL for parallelization
    )
    study.optimize(
        decorated_objective,
        n_trials=n_trials_per_worker,
        show_progress_bar=False,
        callbacks=[wandbc]
    )

    logger.info(f"Worker {worker_id} finished for model {model_type} with study {study_name}")

def tune_model_parallel(model_type: str, data: pd.DataFrame, metadata: dict, study_name: str, exp_path: str):

    logger.info(f"Starting tuning for model {model_type} with study {study_name}")

    project_name = f"local-sdg-synthesisers-{study_name}" if RUN_LOCAL else f"sagemaker-sdg-synthesisers-{study_name}"

    try:
        optuna.delete_study(study_name=project_name, storage=JournalStorage(JournalFileBackend(file_path=JOURNAL_STORAGE)))
    except:
        pass

    worker_args = []
    for i in range(N_WORKERS):
        worker_args.append((
            i, # worker_id
            model_type,
            data,
            metadata,
            project_name,
            JOURNAL_STORAGE,
            N_TRIALS_PER_WORKER,
            exp_path
        ))

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        executor.map(optuna_worker, *zip(*worker_args)) 

    study = optuna.load_study(
        study_name=project_name,
        storage=JournalStorage(JournalFileBackend(file_path=JOURNAL_STORAGE))
    )

    # objective_value = 'Wasserstein'

    # optim_history = optuna.visualization.plot_optimization_history(study, target_name=objective_value)
    # optim_history.update_layout(title_text=None)
    # param_importance = optuna.visualization.plot_param_importances(study)
    # param_importance.update_layout(title_text=None)

    # trials_df = study.trials_dataframe()
    # timeline_plot = optuna_timeline_plot(trials_df)

    # run = wandb.init(
    #     project=project_name,
    #     name='summary_plots',
    #     tags=['summary_plots'],
    # )
    # run.log({
    #     "Optimisation History": optim_history,
    #     "Parameter Importance": param_importance,
    #     "Timeline": timeline_plot,
    # })
    # run.finish()

    best_params = study.best_trial.user_attrs["best_params"]

    return best_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='The name of the dataset (without extension).')
    parser.add_argument('--model', type=str, required=True, help='Name of the model (synthesiser) to tune.')

    args = parser.parse_args()

    dataset = args.dataset
    model = args.model

    train, val, test = load_dataset(dataset)
    train_val = pd.concat([train, val], axis=0, ignore_index=True)

    metadata_path = os.path.join(DATASETS_PATH, f"{dataset}/metadata.toml")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}. Please generate metadata first.")
    metadata = load_config(metadata_path)

    study_name = f"{model}-{dataset}"
    exp_path = os.path.join(EXP_PATH, f'{dataset}/{model}')
    best_params = tune_model_parallel(model, train_val, metadata, study_name, exp_path)

    # save parameters
    params_path = os.path.join(PARAMS_PATH, f'synthesisers/{model}/{dataset}.toml')
    os.makedirs(os.path.dirname(params_path), exist_ok=True)

    dump_config(best_params, params_path)
