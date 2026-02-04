import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import logging
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.single_table import CopulaGANSynthesizer
import torch
from tqdm.auto import tqdm

import wandb

from lib.eval_helper import normalize_data, nearest_neighbors
from lib.preprocess import clean
from lib.utils import dump_json, load_config
from synthesisers.ctgan import init_model as ctgan_init
from synthesisers.tvae import init_model as tvae_init
from synthesisers.gaussian_copula import init_model as gaussian_copula_init
from synthesisers.copula_gan import init_model as copula_gan_init
from synthesisers.binary_diffusion import train as train_bdt
# from synthesisers.binary_diffusion import sample as sample_bdt

DATASETS_PATH = os.getenv("DATASETS_PATH", "/opt/ml/input/data/datasets")
PARAMS_PATH = os.getenv("PARAMS_PATH", "/opt/ml/output/data/params")
EXP_PATH = os.getenv("EXP_PATH", "/opt/ml/output/data/exp")
N_WORKERS_TRAIN = int(os.getenv("N_WORKERS_TRAIN", 2))
N_WORKERS_SAMPLE = int(os.getenv("N_WORKERS", 2))
RUN_LOCAL = os.getenv("RUN_LOCAL", "false").lower() == "true"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - [Proc:%(process)d] - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

logger = logging.getLogger(__name__)

wandb.login()


def train_model(shadow_data, model_params, model_name, shadow_dir):

    metadata = Metadata.detect_from_dataframe(shadow_data)

    if model_name == "ctgan":
        model = ctgan_init(metadata, model_params)
    elif model_name == "tvae":
        model = tvae_init(metadata, model_params)
    elif model_name == "gaussian_copula":
        model = gaussian_copula_init(metadata, model_params)
        # model = GaussianCopulaSynthesizer(metadata)
    elif model_name == "copula_gan":
        model = copula_gan_init(metadata, model_params)
        # model_params.pop('numerical_distributions', None)
        # model = CopulaGANSynthesizer(metadata, **model_params)

    model.fit(shadow_data)

    # save the model
    os.makedirs(shadow_dir, exist_ok=True)
    torch.save(model, os.path.join(shadow_dir, f"{model_name}.pt"))


def train_shadow_model(args, len_real_data, model_params, metadata, model, privacy_dir):

    shadow_id, shadow_data_pd, cur_member = args

    # perpare saved dir
    cur_shadow_dir = os.path.join(privacy_dir, str(shadow_id))
    os.makedirs(cur_shadow_dir, exist_ok=True)
    
    # print(f"{shadow_id} shadow data size: {len(shadow_data_pd)}/{len_real_data}")

    # train model 
    if model != "binary_diffusion":
        # start_time = time.time()
        train_model(shadow_data_pd, model_params, model, cur_shadow_dir)
        # print(f"Shadow model {shadow_id} trained in {time.time() - start_time:.2f} seconds.")
    else: 
        dataset = privacy_dir.split('/')[-3]
        project_name = f"local-sdg-synthesisers-{model}-{dataset}-privacy" if RUN_LOCAL else f"sagemaker-sdg-synthesisers-{model}-{dataset}-privacy"
        wandb.init(project=project_name, name=f'trial {shadow_id}', config=model_params)

        train_bdt(shadow_data_pd, model_params, metadata, cur_shadow_dir)

        wandb.finish()

    # save member info
    with open(os.path.join(cur_shadow_dir, "member.pkl"), "wb") as f:
        pickle.dump(cur_member, f)


def train_shadow_models_parallel(real_data, model_params, metadata, m_shadow_models, model, privacy_dir):
    """
    Train shadow models and save them. 
    """
    size = len(real_data)
    np.random.seed(0)
    keep = np.random.uniform(0, 1, size=(m_shadow_models, size))
    order = keep.argsort(0)
    keep = order < int(0.5 * m_shadow_models)

    # Prepare arguments for parallel execution
    # membership info
    shadow_data_pd = []
    cur_members = []
    for shadow_id in range(m_shadow_models):
        cur_keep = np.array(keep[shadow_id], dtype=bool)
        cur_member = cur_keep.nonzero()[0]
        cur_members.append(cur_member)

        # select data for this shadow model
        shadow_data_pd.append(real_data.iloc[cur_member].reset_index(drop=True))

    len_real_data = len(real_data)

    train_shadow_model_args = partial(
        train_shadow_model,
        len_real_data=len_real_data,
        model_params=model_params,
        metadata=metadata,
        model=model,
        privacy_dir=privacy_dir
    )

    with ProcessPoolExecutor(max_workers=N_WORKERS_TRAIN) as executor:
        results = list(tqdm(
            executor.map(train_shadow_model_args, zip(range(m_shadow_models), shadow_data_pd, cur_members)),
            total=m_shadow_models,
            desc="Training shadow models"
        ))  

from collections import OrderedDict
from functools import partial
from pathlib import Path
import torch
import torch.nn as nn
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
    config["device"] = "cuda" # else "cpu"                            # Device to use for sampling, either 'cuda' or 'cpu'
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

# def sample_bdt(ckpt_path: str, ckpt_transformation: str, n_samples: int, out_path: str = None, seed: int = None):
#     """Sample synthetic data from a pre-trained binary diffusion model.
#     Args:
#         ckpt: Path to the checkpoint file of the pre-trained model.
#         ckpt_transformation: Path to the checkpoint file of the transformation model.
#         n_timesteps: Number of timesteps for sampling.
#         out: Path to the output folder where samples will be saved.
#         n_samples: Number of samples to generate.
#         batch_size: Batch size for sampling.
#         threshold: Threshold for binarization.
#         strategy: Sampling strategy to use, either 'target' or 'mask'.
#         seed: Random seed for reproducibility.
#         guidance_scale: Guidance scale for classifier-free guidance.
#         target_column_name: Name of the target column in the output DataFrame.
#         use_ema: Whether to use the EMA model for sampling.
#         dropna: Whether to drop rows with NaN values during sampling.
#         """

#     if seed:
#         seed_everything(seed)

#     config = sample_config()
#     config["ckpt"] = ckpt_path
#     config["ckpt_transformation"] = ckpt_transformation
#     config["out"] = out_path
#     config["n_samples"] = n_samples
#     config["seed"] = seed

#     if out_path:
#         path_out = Path(config["out"])
#         path_out.mkdir(parents=True, exist_ok=True)

#     ckpt = torch.load(config["ckpt"])
#     device = config["device"]
#     batch_size = int(config["batch_size"])
#     guidance_scale = config["guidance_scale"]
#     threshold = config["threshold"]
#     strategy = config["strategy"]
#     target_column_name = config["target_column_name"]

#     denoising_model = SimpleTableGenerator.from_config(ckpt["config_model"]).to(device)
#     denoising_model.eval()

#     diffusion = BinaryDiffusion1D.from_config(
#         denoise_model=denoising_model,
#         config=ckpt["config_diffusion"],
#     ).to(device)
#     diffusion.eval()

#     transformation = FixedSizeBinaryTableTransformation.from_checkpoint(config["ckpt_transformation"])

#     if config["use_ema"]:
#         ema_state_dict = OrderedDict()
#         for name, param in ckpt["diffusion_ema"].items():
#             if any(substring in name for substring in ['online_model', 'num_batches_tracked', "initted", "step"]):
#                 pass # remove from dict
#             elif 'ema_model' in name:
#                 key = name.replace("ema_model.", "")
#                 ema_state_dict[key] = param
#             else:
#                 ema_state_dict[name] = param

#         diffusion.load_state_dict(ema_state_dict)
#     else:
#         diffusion.load_state_dict(ckpt["diffusion"])

#     n_total_timesteps = diffusion.n_timesteps
#     # timesteps_sampling = select_equally_distributed_numbers(
#     #     n_total_timesteps,
#     #     config["n_timesteps"],
#     # )
#     timesteps_sampling = None # will be set to list(range(5)) in diffusion
#     task = denoising_model.task
#     conditional = denoising_model.conditional
#     n_classes = denoising_model.n_classes
#     classifier_free_guidance = denoising_model.classifier_free_guidance

#     n_generated = 0
#     n_samples = config["n_samples"]
#     pbar = tqdm(total=n_samples)
#     dfs = []

#     while n_generated < n_samples:
#         labels = get_random_labels(
#             conditional=conditional,
#             task=task,
#             n_classes=n_classes,
#             classifier_free_guidance=classifier_free_guidance,
#             n_labels=batch_size,
#             device=device,
#         )

#         x = diffusion.sample(
#             model_fn=(
#                 partial(cfg_model_fn, guidance_scale=guidance_scale, task=task)
#                 if classifier_free_guidance and guidance_scale > 0
#                 else None
#             ),
#             n=batch_size,
#             y=labels,
#             timesteps=timesteps_sampling,
#             threshold=threshold,
#             strategy=strategy,
#         )

#         if conditional:
#             if classifier_free_guidance:
#                 labels = torch.argmax(labels, dim=1)

#             x_df, labels_df = transformation.inverse_transform(x, labels)
#             x_df[target_column_name] = labels_df
#         else:
#             x_df = transformation.inverse_transform(x)

#         if config["dropna"]:
#             x_df = x_df.dropna()

#         n_generated += len(x_df)
#         pbar.update(len(x_df))
#         dfs.append(x_df)

#     df = pd.concat(dfs)

#     if out_path:
#         df.to_csv(path_out, index=False) 

#     pbar.close()

#     return df

def compute_distances_for_shadow_model(shadow_id, real_data, discrete_cols, n_syn_dataset, model_name, privacy_dir):
    """
    Computes distances for a single shadow model and returns the results.
    This function will be executed by each worker process.
    """
    cur_shadow_dir = os.path.join(privacy_dir, str(shadow_id))
    cur_shadow_dist = {}

    if model_name != "binary_diffusion":
        model = torch.load(os.path.join(cur_shadow_dir, f"{model_name}.pt"))
    else:
        ckpt_path = os.path.join(cur_shadow_dir, "model-final-train.pt")
        ckpt_transformation = os.path.join(cur_shadow_dir, "transformation.joblib")
        n_samples = len(real_data) // 2
        config = sample_config()
        config["ckpt"] = ckpt_path
        config["ckpt_transformation"] = ckpt_transformation
        # config["out"] = out_path
        config["n_samples"] = n_samples

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

    for id in range(n_syn_dataset):

        if model_name != "binary_diffusion":
            # start_time = time.time()
            syn_data_pd = model.sample(len(real_data) // 2)
            # print(f"Sampling took {time.time() - start_time:.2f} seconds for shadow model {shadow_id}, dataset {id}.")
        else:
            # syn_data_pd = pd.read_csv(os.path.join(cur_shadow_dir, "sample.csv"))
            # ckpt = os.path.join(cur_shadow_dir, "model-final-train.pt")
            # ckpt_transformation = os.path.join(cur_shadow_dir, "transformation.joblib")
            seed = id
            config["seed"] = seed
            seed_everything(seed)

            # syn_data_pd = sample_bdt(ckpt, ckpt_transformation, len(real_data) // 2, seed=id) 
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

            syn_data_pd = pd.concat(dfs)

            pbar.close()

        clean(syn_data_pd)

        raw_data_arr, syn_data_arr, n_features = normalize_data(real_data, syn_data_pd, discrete_cols)

        distances = nearest_neighbors(syn_data_arr, raw_data_arr)
    
        for i, dist in enumerate(distances):
            normalized_dist = dist[0] / np.sqrt(n_features)
            if i not in cur_shadow_dist:
                cur_shadow_dist[i] = [normalized_dist]
            else:
                cur_shadow_dist[i].append(normalized_dist)

    with open(os.path.join(cur_shadow_dir, "member.pkl"), "rb") as f:
        member = pickle.load(f)

    in_dists = {}
    out_dists = {}
    # get the expected distance for each record
    for id, dist_list in cur_shadow_dist.items():
        mean_dist = np.mean(dist_list)
        if id in member:
            in_dists[id] = [mean_dist]
        else:
            out_dists[id] = [mean_dist]

    # save dists info
    with open(os.path.join(cur_shadow_dir, f"dists.pkl"), "wb") as f:
        pickle.dump((in_dists, out_dists), f)

    return in_dists, out_dists


def compute_MDS_parallel(real_data, m_shadow_models, n_syn_dataset, model_name, discrete_cols, privacy_dir):

    # init the distance dict
    in_member_nbrs_dist = {}
    out_member_nbrs_dist = {}
    for id in range(len(real_data)):
        in_member_nbrs_dist[id] = []
        out_member_nbrs_dist[id] = []

    print("Compute distances for each record in shadow models (in parallel)...")

    # parallise
    func_args = partial(
        compute_distances_for_shadow_model,
        real_data=real_data,
        discrete_cols=discrete_cols,
        n_syn_dataset=n_syn_dataset,
        model_name=model_name,
        privacy_dir=privacy_dir
    )

    with ProcessPoolExecutor(max_workers=N_WORKERS_SAMPLE) as executor:
        results = list(tqdm(
            executor.map(func_args, range(m_shadow_models)),
            total=m_shadow_models,
            desc="Computing distances"
        ))

    # Aggregate results from all processes
    for in_dists, out_dists in results:
        for id, dist_list in in_dists.items():
            in_member_nbrs_dist[id].extend(dist_list)
        for id, dist_list in out_dists.items():
            out_member_nbrs_dist[id].extend(dist_list)
    # end parallise

    # get the DS for each record
    print("Calculate disclosure score for each record...")
    DS = {}
    for id in range(len(real_data)):
        mean_in_dist = np.mean(in_member_nbrs_dist[id])
        mean_out_dist = np.mean(out_member_nbrs_dist[id])
        DS[id] = abs(mean_in_dist - mean_out_dist)

    # get the MDS
    MDS = max(DS.values())

    print("membership disclosure score: {}".format(MDS))

    # save the distance difference
    with open(os.path.join(privacy_dir, "disclosure_score.pkl"), "wb") as f:
        pickle.dump(DS, f)

    return MDS


def evaluate_privacy(real_data, model_params, m_shadow_models, n_syn_dataset, model, metadata, privacy_dir):

    print("Train shadow models...")
    # train_shadow_models_parallel(real_data, model_params, metadata, m_shadow_models, model, privacy_dir)
    print("Shadow models trained.")

    cat_features = metadata['cat_features']
    discrete_cols = cat_features + [metadata["target_column"]] if metadata["task"] == "classification" else cat_features

    print("Compute MDS...")
    MDS = compute_MDS_parallel(real_data, m_shadow_models, n_syn_dataset, model, discrete_cols, privacy_dir)

    return {"MDS": MDS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='The name of the dataset (without extension).')
    parser.add_argument('--model', type=str, help='Name of the model (synthesiser) to evaluate privacy for.')
    parser.add_argument("--m_shadow_models", "-m_model", type=int, default=20)
    parser.add_argument("--n_syn_dataset", "-n_syn", type=int, default=100)

    args = parser.parse_args()

    dataset = args.dataset
    model = args.model

    dataset_path = os.path.join(DATASETS_PATH, f"{dataset}/{dataset}.csv")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}. Please provide a valid dataset path.")
    
    df = pd.read_csv(dataset_path)

    clean(df)

    params_path = os.path.join(PARAMS_PATH, f'synthesisers/{model}/{dataset}.toml')

    if not os.path.exists(params_path):
        logger.info(f"Parameters for synthesiser not found at {params_path}.")
    
        # Attempt to find the parameters in the input directory
        params_path = params_path.replace('output', 'input')

        logger.info(f"Attempting to load parameters from {params_path}")

        if not os.path.exists(params_path):
            raise FileNotFoundError(f"Parameters file not found at {params_path}. Please generate parameters first or move them to input/data/exp/dataset/model/synthesisers.")

    model_params = load_config(params_path)

    metadata_path = os.path.join(DATASETS_PATH, f"{dataset}/metadata.toml")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}. Please generate metadata first.")

    metadata = load_config(metadata_path)
    
    privacy_path = os.path.join(EXP_PATH, f'{dataset}/{model}/privacy')

    privacy_path = privacy_path.replace('output', 'input')

    privacy_result = evaluate_privacy(df, model_params, args.m_shadow_models, args.n_syn_dataset, model, metadata, privacy_path)

    results_path = os.path.join(os.path.join(EXP_PATH, f'{dataset}/{model}/privacy'), f'privacy_result.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    dump_json(privacy_result, results_path)


if __name__ == "__main__":
    main()