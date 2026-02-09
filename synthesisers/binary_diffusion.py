import argparse
from collections import OrderedDict
import os 
from pathlib import Path
from functools import partial

import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn

import wandb

from synthesisers.binary_diffusion_tabular import (
    BinaryDiffusion1D,
    SimpleTableGenerator,
    FixedSizeBinaryTableDataset,
    FixedSizeBinaryTableTransformation,
    FixedSizeTableBinaryDiffusionTrainer,
    select_equally_distributed_numbers,
    TASK,
    get_random_labels,
    drop_fill_na,
    seed_everything
)

from lib.evaluation import cal_fidelity
from lib.preprocess import clean

BD_TRIALS = int(os.getenv("BD_TRIALS", 20))  # Number of trials for Binary Diffusion when tuned sequentially
BD_TRAIN_STEPS = int(os.getenv("BD_TRAIN_STEPS", 200000))  # Number of training steps for Binary Diffusion
BD_TUNE_STEPS = int(os.getenv("BD_TUNE_STEPS", 50000))  # Number of tuning steps for Binary Diffusion
STORAGE = os.getenv("STORAGE", "sqlite:////opt/ml/output/data/optuna_study.db")
N_DATALOADERS = int(os.getenv("N_DATALOADERS", os.cpu_count() - 2)) # number of dataloaders - assumes only one process. 2 less than the number of CPUs
RUN_LOCAL = os.getenv("RUN_LOCAL", "false").lower() == "true"


def train_config(params, metadata, tune=False):

    config = {}
    config["fine_tune_from"] = None
    # config["comment"] = "dataset_name"

    data = {}
    data["path_table"] = None                               # Dont need if i already have dataset

    data["numerical_columns"] = metadata["num_features"]
    data["categorical_columns"] = metadata["cat_features"]
    data["target_column"] = None                            # No target column as we are training unconditionally
    data["task"] = metadata["task"]

    data["columns_to_drop"] = None
    data["dropna"] = True
    data["fillna"] = False
    data["target_column"] = None                            # unconditional generation - always
    data["split_feature_target"] = False                    # No split as we are training unconditionally

    config["data"] = data

    model = {}                   # doesnt change
    model["dim"] = 256                               
    model["n_res_blocks"] = 3                       

    config["model"] = model

    diffusion = {}               # doesnt change
    diffusion["schedule"] = "quad"
    diffusion["n_timesteps"] = 1000
    diffusion["target"] = "two_way"

    config["diffusion"] = diffusion

    trainer = {}
    trainer["log_every"] = 100
    trainer["max_grad_norm"] = None
    trainer["gradient_accumulate_every"] = 1
    trainer["dataloader_workers"] = N_DATALOADERS
    trainer["classifier_free_guidance"] = False
    trainer["zero_token_probability"] = 0.1 
    trainer["ema_decay"] = 0.995    
    trainer["ema_update_every"] = 10
    trainer["opt_type"] = "adam"
    trainer["opt_params"] = None
    trainer["save_num_samples"] = 64

    if tune:
        trainer["train_num_steps"] = BD_TUNE_STEPS
        trainer["save_every"] = 100000 
        # To ensure samples are not saved during tuning, set save_every to 
        # something larger than num_train_steps. 
        trainer["lr"] = params["lr"]
        trainer["batch_size"] = params["batch_size"]

    else: 
        trainer["train_num_steps"] = BD_TRAIN_STEPS
        trainer["save_every"] = 10000 # save often in case training job fails        

        trainer["lr"] = params["lr"] # default = 1e-4
        trainer["batch_size"] = params["batch_size"] # default = 256
    

    config["trainer"] = trainer

    return config


def train(data: pd.DataFrame, model_params: dict, metadata: dict, base_out_path: str, tune=False, worker_id=0, seed=42):
    """Train a binary diffusion synthesiser on the given data.

    Args:
        data (pd.DataFrame): The training data.
        model_params (dict): Parameters for the model.
        metadata (dict): Metadata of the dataset.
        base_out_path (str): Base path for saving outputs.
        sample (bool, optional): Whether to sample synthetic data after training. Defaults to True.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
    """
    if seed:
        seed_everything(seed)

    if tune:
        config = train_config(model_params, metadata, tune=True)
        config["trainer"]["save_num_samples"] = len(data) # if len(data) < 10000 else 10000  # If dataset is smaller than 10k, save all samples, otherwise save 10k samples
    else:
        config = train_config(model_params, metadata)
        config["trainer"]["save_num_samples"] = len(data) # if len(data) < 10000 else 10000

    config["result_folder"] = Path(base_out_path)
    
    device = "cuda"

    # dont need to call drop_fill_na as we already have a clean dataset

    dataset = FixedSizeBinaryTableDataset(
        table=data,
        target_column=config["data"]["target_column"], 
        split_feature_target=config["data"]["split_feature_target"], 
        task=config["data"]["task"],
        numerical_columns=config["data"]["numerical_columns"],
        categorical_columns=config["data"]["categorical_columns"],
    )

    model = SimpleTableGenerator(
        data_dim=dataset.row_size,
        out_dim=(
            dataset.row_size * 2
            if config["diffusion"]["target"] == "two_way"
            else dataset.row_size
        ),
        task=dataset.task,
        conditional=dataset.conditional,
        n_classes=0, # n_classes=0 if config["data"]["task"] == "regression" else dataset.n_classes - if conditional else 0. if not conditional y is not separated from dataframe and label encoder is not fit so le.classes_ attribute DNE
        classifier_free_guidance=config["trainer"]["classifier_free_guidance"],
        **config["model"]
    ).to(device)

    diffusion = BinaryDiffusion1D(
        denoise_model=model,
        **config["diffusion"],
    ).to(device)

    trainer = FixedSizeTableBinaryDiffusionTrainer(
        diffusion=diffusion,
        dataset=dataset,
        **config["trainer"],
        logger=None,
        results_folder=config["result_folder"],
        worker_id=worker_id,
    )

    synth_sample = trainer.train(tune=tune)
    
    if tune:
        return synth_sample

    
def tune(data: pd.DataFrame, metadata: dict, base_out_path: str, study_name: str, seed=42):

    def objective(trial):
        model_params = {}
        model_params["lr"] = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        model_params["batch_size"] = trial.suggest_categorical("batch_size", [64, 128, 256, 512])

        trial.set_user_attr("best_params", model_params)

        synth_sample = train(data, model_params, metadata, base_out_path, tune=True, seed=trial.number + seed)

        clean(synth_sample)

        fidelity_score = cal_fidelity(data, synth_sample, tune=True)

        wandb.log({'fidelity score': fidelity_score})

        return fidelity_score
    
    project_name = f"sagemaker-sdg-synthesisers-{study_name}"
    wandb_kwargs = {"project": project_name}
    wandbc = WeightsAndBiasesCallback(
        metric_name='wasserstein',
        wandb_kwargs=wandb_kwargs,
        as_multirun=True
    )

    decorated_objective = wandbc.track_in_wandb()(objective) # allows wandb.log inside objective function
    
    try:
        optuna.delete_study(study_name=project_name, storage=STORAGE)
    except:
        pass

    study = optuna.create_study(
        direction='minimize',
        sampler= optuna.samplers.TPESampler(seed=0),
        storage=STORAGE,
        study_name=project_name,
    )

    study.optimize(decorated_objective, n_trials=BD_TRIALS, show_progress_bar=True, callbacks=[wandbc])

    best_params = study.best_trial.user_attrs["best_params"]

    return best_params


def sample_config():
    config = {}
    # config["ckpt"] = "path/to/your/checkpoint.pth"            # Path to checkpoint file
    # config["ckpt_transformation"] = "path/to/jooblib/file"    # Path to transformation checkpoint file
    config["n_timesteps"] = 5                                   # Number of sampling steps. Default is 5.
    # config["out"] = "path/to/out/folder"                      # Path to output folder, where to save samples
    config["n_samples"] = 5000                                  # Number of samples to generate. Default is 5000. overwritten in main sample function
    config["batch_size"] = 256                                  # Batch size for sampling
    config["threshold"] = 0.5                                   # Threshold for binarization
    config["strategy"] = "target"                               # Sampling strategy, either 'target' or 'mask'
    config["seed"] = 42                                         # Random seed for reproducibility
    config["guidance_scale"] = 0.0                              # Guidance scale for classifier-free guidance - must be 0 as model is trained unconditionally
    config["target_column_name"] = "target"                     # Name of the target column
    config["device"] = "cuda"
    config["use_ema"] = True                                    # Whether to use the EMA model for sampling - Default is True, as the model is trained with EMA
    config["dropna"] = False                                    # Whether to drop rows with NaN values during sampling

    return config


def cfg_model_fn(
    x_t: torch.Tensor,
    ts: torch.Tensor,
    y: torch.Tensor,
    model: nn.Module,
    guidance_scale: float,
    task: TASK,
    *args,
    **kwargs
) -> torch.Tensor:
    """Classifier free guidance sampling function

    Args:
        x_t: noisy sample
        ts: timesteps
        y: conditioning
        model: denoising model
        guidance_scale: guidance scale in classifier free guidance
        task: dataset task

    Returns:
        torch.Tensor: denoiser output
    """

    combine = torch.cat([x_t, x_t], dim=0)
    combine_ts = torch.cat([ts, ts], dim=0)

    if task == "classification":
        y_other = torch.zeros_like(y)
    elif task == "regression":
        # for regression, zero-token is -1, since values are minmax normalized to [0, 1] range
        y_other = torch.ones_like(y) * -1

    combine_y = torch.cat([y, y_other], dim=0)
    model_out = model(combine, combine_ts, y=combine_y)
    cond_eps, uncod_eps = torch.split(model_out, [y.shape[0], y.shape[0]], dim=0)
    eps = uncod_eps + guidance_scale * (cond_eps - uncod_eps)
    return eps


def sample(ckpt_path: str, ckpt_transformation: str, n_samples: int, out_path: str = None, seed: int = None):
    """Sample synthetic data from a pre-trained binary diffusion model.
    Args:
        ckpt: Path to the checkpoint file of the pre-trained model.
        ckpt_transformation: Path to the checkpoint file of the transformation model.
        n_timesteps: Number of timesteps for sampling.
        out: Path to the output folder where samples will be saved.
        n_samples: Number of samples to generate.
        batch_size: Batch size for sampling.
        threshold: Threshold for binarization.
        strategy: Sampling strategy to use, either 'target' or 'mask'.
        seed: Random seed for reproducibility.
        guidance_scale: Guidance scale for classifier-free guidance.
        target_column_name: Name of the target column in the output DataFrame.
        use_ema: Whether to use the EMA model for sampling.
        dropna: Whether to drop rows with NaN values during sampling.
        """

    if seed:
        seed_everything(seed)

    config = sample_config()
    config["ckpt"] = ckpt_path
    config["ckpt_transformation"] = ckpt_transformation
    config["out"] = out_path
    config["n_samples"] = n_samples
    config["seed"] = seed

    if out_path:
        path_out = Path(config["out"])
        path_out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(config["ckpt"])
    device = config["device"]
    batch_size = int(config["batch_size"])
    guidance_scale = config["guidance_scale"]
    threshold = config["threshold"]
    strategy = config["strategy"]
    target_column_name = config["target_column_name"]

    denoising_model = SimpleTableGenerator.from_config(ckpt["config_model"]).to(device)
    denoising_model.eval()

    diffusion = BinaryDiffusion1D.from_config(
        denoise_model=denoising_model,
        config=ckpt["config_diffusion"],
    ).to(device)
    diffusion.eval()

    transformation = FixedSizeBinaryTableTransformation.from_checkpoint(config["ckpt_transformation"])

    if config["use_ema"]:
        ema_state_dict = OrderedDict()
        for name, param in ckpt["diffusion_ema"].items():
            if any(substring in name for substring in ['online_model', 'num_batches_tracked', "initted", "step"]):
                pass # remove from dict
            elif 'ema_model' in name:
                key = name.replace("ema_model.", "")
                ema_state_dict[key] = param
            else:
                ema_state_dict[name] = param

        diffusion.load_state_dict(ema_state_dict)
    else:
        diffusion.load_state_dict(ckpt["diffusion"])

    n_total_timesteps = diffusion.n_timesteps
    # timesteps_sampling = select_equally_distributed_numbers(
    #     n_total_timesteps,
    #     config["n_timesteps"],
    # )
    timesteps_sampling = None # will be set to list(range(5)) in diffusion
    task = denoising_model.task
    conditional = denoising_model.conditional
    n_classes = denoising_model.n_classes
    classifier_free_guidance = denoising_model.classifier_free_guidance

    n_generated = 0
    n_samples = config["n_samples"]
    pbar = tqdm(total=n_samples)
    dfs = []

    while n_generated < n_samples:
        labels = get_random_labels(
            conditional=conditional,
            task=task,
            n_classes=n_classes,
            classifier_free_guidance=classifier_free_guidance,
            n_labels=batch_size,
            device=device,
        )

        x = diffusion.sample(
            model_fn=(
                partial(cfg_model_fn, guidance_scale=guidance_scale, task=task)
                if classifier_free_guidance and guidance_scale > 0
                else None
            ),
            n=batch_size,
            y=labels,
            timesteps=timesteps_sampling,
            threshold=threshold,
            strategy=strategy,
        )

        if conditional:
            if classifier_free_guidance:
                labels = torch.argmax(labels, dim=1)

            x_df, labels_df = transformation.inverse_transform(x, labels)
            x_df[target_column_name] = labels_df
        else:
            x_df = transformation.inverse_transform(x)

        if config["dropna"]:
            x_df = x_df.dropna()

        n_generated += len(x_df)
        pbar.update(len(x_df))
        dfs.append(x_df)

    df = pd.concat(dfs)

    if out_path:
        df.to_csv(path_out, index=False) 

    pbar.close()

    return df
