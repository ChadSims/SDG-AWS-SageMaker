import os
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer
import torch
import wandb

from lib.evaluation import cal_fidelity

N_TRIALS_SYNTHESISERS = int(os.getenv("N_TRIALS_SYNTHESISERS", 50))
STORAGE = os.getenv("STORAGE", "sqlite:////opt/ml/output/data/optuna_study.db")


def init_model(metadata: Metadata, model_params):
    """
    Initialize the CTGAN model with the given parameters and metadata.
    """
    model = CTGANSynthesizer(
        metadata,
        embedding_dim=model_params["embedding_dim"],
        generator_dim=model_params["generator_dim"],
        discriminator_dim=model_params["discriminator_dim"],
        generator_lr=model_params["generator_lr"],
        generator_decay=model_params["generator_decay"],
        discriminator_lr=model_params["discriminator_lr"],
        discriminator_decay=model_params["discriminator_decay"],
        batch_size=model_params["batch_size"],
        epochs=model_params["epochs"],
        cuda = torch.cuda.is_available()
    )
    
    return model

def train(data: pd.DataFrame, model_params: dict, base_out_path: str, tune=False, seed=42):
    # improve_reproducibility(seed)
    
    metadata = Metadata.detect_from_dataframe(data)

    model = init_model(metadata, model_params)

    model.fit(data)

    loss = model.get_loss_values()
    synthetic_data = model.sample(len(data))

    if tune:
        return synthetic_data, loss
    else:
        synthetic_data_path = os.path.join(base_out_path, 'sample.csv')
        synthetic_data.to_csv(synthetic_data_path, index=False)
        
        loss_path = os.path.join(base_out_path, 'loss.csv')
        loss.to_csv(loss_path, index=False)

        model_path = os.path.join(base_out_path, 'ctgan.pt')
        torch.save(model, model_path)
        

def tune(train_data: pd.DataFrame, val_data: pd.DataFrame, study_name: str, storage: str = None):
    
    def objective(trial):

        model_params = {}
        model_params["generator_lr"] = trial.suggest_float("generator_lr", 1e-5, 1e-3, log=True)
        model_params["discriminator_lr"] = trial.suggest_float("discriminator_lr", 1e-5, 1e-3, log=True)
        model_params["generator_decay"] = 1e-6
        model_params["discriminator_decay"] = trial.suggest_categorical("discriminator_decay", [0, 1e-6])
        # model_params["batch_size"] = trial.suggest_categorical("batch_size", [40, 80, 160, 320, 500])
        # model_params["epochs"] = trial.suggest_int("epochs", 100, 500)

        model_params["batch_size"] = 500 # affects runtime more than fidelity -> higher batch size is better
        model_params["epochs"] = 250  # Fixed for tuning, can be adjusted later
        model_params["embedding_dim"] = 128
        model_params["generator_dim"] = [128, 128]
        model_params["discriminator_dim"] = [256, 256]

        trial.set_user_attr("best_params", model_params)

        synthetic_data, loss = train(data, model_params, None, tune=True)

        loss["trial"] = trial.number

        loss_table = wandb.Table(dataframe=loss)
        wandb.log(
            {
                f"ctgan-loss-tune": loss_table,
            }
        )

        fidelity_score = cal_fidelity(data, synthetic_data, tune=True)

        return fidelity_score
    
    data = pd.concat([train_data, val_data], ignore_index=True, sort=False)
    metadata = Metadata.detect_from_dataframe(data)

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

    study.optimize(decorated_objective, n_trials=N_TRIALS_SYNTHESISERS, show_progress_bar=True,  callbacks=[wandbc])

    best_params = study.best_trial.user_attrs["best_params"]

    return best_params


def sample(model_path: str, n_samples: int):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = torch.load(model_path)
    synthetic_data = model.sample(n_samples)

    return synthetic_data