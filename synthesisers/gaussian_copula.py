import os
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer
import torch

from lib.evaluation import cal_fidelity

N_TRIALS_SYNTHESISERS = int(os.getenv("N_TRIALS_SYNTHESISERS", 50))
STORAGE = os.getenv("STORAGE", "sqlite:////opt/ml/output/data/optuna_study.db")


def init_model(metadata: Metadata, model_params):
    
    model = GaussianCopulaSynthesizer(
        metadata,
        numerical_distributions=model_params["numerical_distributions"],
    )

    return model


def train(data: pd.DataFrame, model_params: dict, base_out_path: str, tune=False, seed=42):

    metadata = Metadata.detect_from_dataframe(data)

    model = init_model(metadata, model_params)

    model.fit(data)

    synthetic_data = model.sample(len(data))

    if tune:
        return synthetic_data
    else:
        synthetic_data_path = os.path.join(base_out_path, 'sample.csv')
        synthetic_data.to_csv(synthetic_data_path, index=False)

        model_path = os.path.join(base_out_path, 'gaussian_copula.pt')
        torch.save(model, model_path)


def tune(train_data: pd.DataFrame, val_data: pd.DataFrame, study_name: str, storage: str = None):
    
    def objective(trial):
        model_params = {}
        model_params["numerical_distributions"] = {}

        for col in num_features:
            model_params["numerical_distributions"][col] = trial.suggest_categorical(
                f"numerical_distribution_{col}",
                ["norm", "truncnorm", "uniform", "gamma"] # "beta" and "gaussian_kde" removed: "beta" can fail sometimes, "gaussian_kde" slows optimisation significantly
            )

        trial.set_user_attr("best_params", model_params)

        model = init_model(metadata, model_params)
        model.fit(data)

        synthetic_data = model.sample(len(data))

        fidelity_score = cal_fidelity(data, synthetic_data, tune=True)

        return fidelity_score


    data = pd.concat([train_data, val_data], ignore_index=True, sort=False)
    metadata = Metadata.detect_from_dataframe(data)

    sdtypes = {col: metadata.to_dict()['tables']['table']['columns'][col]['sdtype'] for col in data.columns}
    num_features = [col for col, sdtype in sdtypes.items() if sdtype == 'numerical']

    project_name = f"sagemaker-sdg-synthesisers-{study_name}"
    wandb_kwargs = {"project": project_name}
    wandbc = WeightsAndBiasesCallback(
        metric_name='wasserstein',
        wandb_kwargs=wandb_kwargs,
        as_multirun=True
    )

    decorated_objective = wandbc.track_in_wandb()(objective) # allows wandb.log inside objective function
    
    if storage is None:
        storage = STORAGE

    try:
        optuna.delete_study(study_name=project_name, storage=storage)
    except:
        pass

    study = optuna.create_study(
        direction='minimize',
        sampler= optuna.samplers.TPESampler(seed=0),
        storage=storage,
        study_name=project_name,
    )

    study.optimize(decorated_objective, n_trials=N_TRIALS_SYNTHESISERS, show_progress_bar=True)

    best_params = study.best_trial.user_attrs["best_params"]

    return best_params


def sample(model_path: str, n_samples: int):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = torch.load(model_path)
    synthetic_data = model.sample(n_samples)
    
    return synthetic_data