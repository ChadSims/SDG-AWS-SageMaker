import argparse
import gc
import logging
import multiprocessing
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import pandas as pd
import torch
import wandb
from tqdm.auto import tqdm
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.single_table import CopulaGANSynthesizer

from lib.eval_helper import normalize_data, nearest_neighbors
from lib.preprocess import clean
from lib.utils import dump_json, load_config
from synthesisers.ctgan import init_model as ctgan_init
from synthesisers.tvae import init_model as tvae_init
from synthesisers.gaussian_copula import init_model as gaussian_copula_init
from synthesisers.copula_gan import init_model as copula_gan_init
from synthesisers.binary_diffusion import train as train_bdt
from synthesisers.binary_diffusion import sample as sample_bdt

# Environment Configuration
DATASETS_PATH = os.getenv("DATASETS_PATH", "/opt/ml/input/data/datasets")
PARAMS_PATH = os.getenv("PARAMS_PATH", "/opt/ml/output/data/params")
EXP_PATH = os.getenv("EXP_PATH", "/opt/ml/output/data/exp")
N_WORKERS_TRAIN = int(os.getenv("N_WORKERS_TRAIN", 2))
N_WORKERS_SAMPLE = int(os.getenv("N_WORKERS", 2))
M_SHADOW_MODELS = int(os.getenv("M_SHADOW_MODELS", 20))
N_SYN_DATASET = int(os.getenv("N_SYN_DATASET", 100))
RUN_LOCAL = os.getenv("RUN_LOCAL", "false").lower() == "true"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - [Proc:%(process)d] - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

logger = logging.getLogger(__name__)

wandb.login()


def cleanup_resources():
    """Universal cleanup for both CPU and GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
    gc.collect()


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
        # model = copula_gan_init(metadata, model_params)
        model_params.pop('numerical_distributions', None)
        model_params["default_distribution"] = "gamma"
        model = CopulaGANSynthesizer(metadata, **model_params)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.fit(shadow_data)

    # Move to CPU before saving to prevent GPU affinity issues during load
    if hasattr(model, '_model') and hasattr(model._model, 'cpu'):
        model._model = model._model.cpu()

    # save the model
    os.makedirs(shadow_dir, exist_ok=True)
    torch.save(model, os.path.join(shadow_dir, f"{model_name}.pt"))
    del model
    cleanup_resources()


def shadow_worker_task(args, real_data_ref, model_params, metadata, model_name, privacy_dir):

    shadow_id, member_indices = args
    cur_shadow_dir = os.path.join(privacy_dir, str(shadow_id))
    os.makedirs(cur_shadow_dir, exist_ok=True)
    
    shadow_data_pd = real_data_ref.iloc[member_indices].reset_index(drop=True)

    try:
        # train model 
        if model_name != "binary_diffusion":
            train_model(shadow_data_pd, model_params, model_name, cur_shadow_dir)
        else: 
            dataset_name = privacy_dir.split('/')[-3]
            project_name = f"{'local' if RUN_LOCAL else 'sagemaker'}-sdg-{model_name}-{dataset_name}-privacy"
            wandb.init(project=project_name, name=f'trial {shadow_id}', config=model_params, reinit=True)
            train_bdt(shadow_data_pd, model_params, metadata, cur_shadow_dir)
            wandb.finish()

        # save member info
        with open(os.path.join(cur_shadow_dir, "member.pkl"), "wb") as f:
            pickle.dump(member_indices, f)
    
    finally:
        cleanup_resources()


def train_shadow_models_parallel(real_data, model_params, metadata, m_shadow_models, model, privacy_dir):
    """
    Train shadow models and save them. 
    """
    size = len(real_data)
    np.random.seed(0)

    keep = np.random.uniform(0, 1, size=(m_shadow_models, size))
    order = keep.argsort(0)
    final_mask = order < int(0.5 * m_shadow_models)

    # Prepare index lists 
    task_args = []
    for shadow_id in range(m_shadow_models):
        member_indices = np.where(final_mask[shadow_id])[0]
        task_args.append((shadow_id, member_indices))

    worker_fn = partial(
        shadow_worker_task,
        real_data_ref=real_data,
        model_params=model_params,
        metadata=metadata,
        model_name=model,
        privacy_dir=privacy_dir
    )

    with ProcessPoolExecutor(max_workers=N_WORKERS_TRAIN) as executor:
        list(tqdm(
            executor.map(worker_fn, task_args),
            total=m_shadow_models, 
            desc="Training Shadow Models"
        ))


def compute_distances_worker(shadow_id, real_data, discrete_cols, n_syn_dataset, model_name, privacy_dir):
    """
    Computes distances for a single shadow model and returns the results.
    This function will be executed by each worker process.
    """
    cur_shadow_dir = os.path.join(privacy_dir, str(shadow_id))
    
    # Pre-calculate sqrt(n_features) for normalization
    n_features = real_data.shape[1]
    norm_factor = np.sqrt(n_features)

    model = None
    if model_name != "binary_diffusion":
        model = torch.load(os.path.join(cur_shadow_dir, f"{model_name}.pt"), map_location='cpu')

    # Result structure: {record_id: [distances_from_N_samples]}
    temp_dist_accumulator = {i: [] for i in range(len(real_data))}

    try:
        for _ in range(n_syn_dataset):

            if model_name != "binary_diffusion":
                syn_data_pd = model.sample(len(real_data) // 2)
            else:
                ckpt = os.path.join(cur_shadow_dir, "model-final-train.pt")
                ckpt_transformation = os.path.join(cur_shadow_dir, "transformation.joblib")
                syn_data_pd = sample_bdt(ckpt, ckpt_transformation, len(real_data) // 2) 

            clean(syn_data_pd)

            raw_data_arr, syn_data_arr, _ = normalize_data(real_data, syn_data_pd, discrete_cols)

            distances = nearest_neighbors(syn_data_arr, raw_data_arr)

            for i, dist in enumerate(distances):
                temp_dist_accumulator[i].append(dist[0] / norm_factor)

        with open(os.path.join(cur_shadow_dir, "member.pkl"), "rb") as f:
            member_indices = set(pickle.load(f))

        in_dists = {}
        out_dists = {}
        for idx, dist_list in temp_dist_accumulator.items():
            avg_dist = np.mean(dist_list)
            if idx in member_indices:
                in_dists[idx] = avg_dist
            else:
                out_dists[idx] = avg_dist

        # save dists info
        with open(os.path.join(cur_shadow_dir, f"dists.pkl"), "wb") as f:
            pickle.dump((in_dists, out_dists), f)

        return in_dists, out_dists
    
    finally:
        del model
        cleanup_resources()


def compute_MDS_parallel(real_data, m_shadow_models, n_syn_dataset, model_name, discrete_cols, privacy_dir):

    # Use 2D arrays for high performance: [record_id, shadow_model_id]
    # Initialize with NaNs to differentiate from actual distances
    in_matrix = np.full((len(real_data), m_shadow_models), np.nan)
    out_matrix = np.full((len(real_data), m_shadow_models), np.nan)

    worker_fn = partial(
        compute_distances_worker, 
        real_data=real_data, 
        discrete_cols=discrete_cols,
        n_syn_dataset=n_syn_dataset,
        model_name=model_name,
        privacy_dir=privacy_dir
    )

    with ProcessPoolExecutor(max_workers=N_WORKERS_SAMPLE) as executor:
        results = list(tqdm(
            executor.map(worker_fn, range(m_shadow_models)),
            total=m_shadow_models,
            desc="Computing Distances"
        ))

    # Aggregate results from all processes
    for s_id, (in_dists, out_dists) in enumerate(results):
        for r_id, val in in_dists.items(): in_matrix[r_id, s_id] = val
        for r_id, val in out_dists.items(): out_matrix[r_id, s_id] = val

    # get the DS for each record
    mean_in = np.nanmean(in_matrix, axis=1)
    mean_out = np.nanmean(out_matrix, axis=1)
    DS = {i: float(abs(mean_in[i] - mean_out[i])) for i in range(len(real_data))}

    # get the MDS
    MDS = max(DS.values())
    logger.info(f"Membership Disclosure Score: {MDS}")

    # save the distance difference
    with open(os.path.join(privacy_dir, "disclosure_score.pkl"), "wb") as f:
        pickle.dump(DS, f)

    return MDS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='The name of the dataset (without extension).')
    parser.add_argument('--model', type=str, required=True, help='Name of the model (synthesiser) to evaluate privacy for.')
    args = parser.parse_args()

    dataset_path = os.path.join(DATASETS_PATH, f"{args.dataset}/{args.dataset}.csv")
    df = pd.read_csv(dataset_path)
    clean(df)

    metadata_path = os.path.join(DATASETS_PATH, f"{args.dataset}/metadata.toml")
    metadata = load_config(metadata_path)

    params_path = os.path.join(PARAMS_PATH, f'synthesisers/{args.model}/{args.dataset}.toml')
    if not os.path.exists(params_path):
        # Attempt to find the parameters in the input directory
        params_path = params_path.replace('output', 'input')

    model_params = load_config(params_path)

    privacy_path = os.path.join(EXP_PATH, f'{args.dataset}/{args.model}/privacy')
    os.makedirs(privacy_path, exist_ok=True)

    # Evaluation pipeline
    train_shadow_models_parallel(df, model_params, metadata, M_SHADOW_MODELS, args.model, privacy_path)

    cleanup_resources()

    cat_features = metadata['cat_features']
    discrete_cols = cat_features + [metadata["target_column"]] if metadata["task"] == "classification" else cat_features

    mds_val = compute_MDS_parallel(df, M_SHADOW_MODELS, N_SYN_DATASET, args.model, discrete_cols, privacy_path)

    dump_json({"MDS": mds_val}, os.path.join(privacy_path, f'privacy_result.json'))


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()