import os

import optuna
import pandas as pd
from optuna.integration.wandb import WeightsAndBiasesCallback

import wandb
from lib.evaluation import cal_fidelity
from synthesisers.potnet_core import POTNet, load_model

N_TRIALS_SYNTHESISERS = int(os.getenv("N_TRIALS_SYNTHESISERS", str(50)))
STORAGE = os.getenv("STORAGE", "sqlite:////opt/ml/output/data/optuna_study.db")
RUN_LOCAL = os.getenv("RUN_LOCAL", "false").lower() == "true"


def init_model(model_params: dict):

    potnet_model = POTNet(
        embedding_dim=model_params["embedding_dim"],
        categorical_cols=model_params["categorical_cols"],
        numeric_output_data_type=model_params["numeric_output_data_type"],
        epochs=model_params["epochs"],
        batch_size=model_params["batch_size"],
        lr=model_params["lr"],
    )

    return potnet_model


def train(data: pd.DataFrame, model_params: dict, base_out_path: str, tune=False):

    potnet_model = init_model(model_params=model_params)

    potnet_model.fit(data)

    synthetic_data = potnet_model.generate(len(data))

    if tune:
        return synthetic_data
    else:
        synthetic_data_path = os.path.join(base_out_path, "sample.csv")
        synthetic_data.to_csv(synthetic_data_path, index=False)

        model_path = os.path.join(base_out_path, "potnet.pt")
        potnet_model.save(model_path=model_path)


def tune(data: pd.DataFrame, metadata: dict, study_name: str) -> dict:

    def objective(trial) -> float:
        model_params = {}

        model_params["embedding_dim"] = data.shape[1]

        cat_features = metadata["cat_features"]
        num_features = metadata["num_features"]
        if metadata["task"] == "regression":
            num_features.append(metadata["target_column"])
        else:
            cat_features.append(metadata["target_column"])
        int_cols = [col for col in num_features if data[col][0].dtype == "int64"]
        cont_col = [col for col in num_features if data[col][0].dtype == "float64"]
        numeric_output_data_type = {"integer": int_cols, "continuous": cont_col}
        model_params["numeric_output_data_type"] = numeric_output_data_type
        model_params["categorical_cols"] = cat_features
        model_params["epochs"] = 300

        model_params["batch_size"] = trial.suggest_categorical(
            "batch_size", [32, 64, 128, 256, 512]
        )
        model_params["lr"] = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

        trial.set_user_attr("params", model_params)

        synth_sample = train(data, model_params, None, tune=True)

        return cal_fidelity(data, synth_sample, tune=True)

    project_name = (
        f"{'local' if RUN_LOCAL else 'sagemaker'}-sdg-synthesisers-{study_name}"
    )
    wandb_kwargs = {"project": project_name}
    wandbc = WeightsAndBiasesCallback(
        metric_name="wasserstein", wandb_kwargs=wandb_kwargs, as_multirun=True
    )

    decorated_objective = wandbc.track_in_wandb()(
        objective
    )  # allows wandb.log inside objective function

    try:
        optuna.delete_study(study_name=project_name, storage=STORAGE)
    except:
        pass

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=0),
        storage=STORAGE,
        study_name=project_name,
    )

    study.optimize(
        decorated_objective, n_trials=N_TRIALS_SYNTHESISERS, show_progress_bar=True
    )

    best_params = study.best_trial.user_attrs["params"]

    return best_params


def sample(model_path: str, n_samples: int):
    potnet_model = load_model(model_path=model_path)
    synthetic_data = potnet_model.generate(n_samples)

    return synthetic_data
