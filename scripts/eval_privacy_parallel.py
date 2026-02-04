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
from synthesisers.binary_diffusion import sample as sample_bdt

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
        # model = copula_gan_init(metadata, model_params)
        model_params.pop('numerical_distributions', None)
        model_params["default_distribution"] = "gamma"
        model = CopulaGANSynthesizer(metadata, **model_params)

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


def compute_distances_for_shadow_model(shadow_id, real_data, discrete_cols, n_syn_dataset, model_name, privacy_dir):
    """
    Computes distances for a single shadow model and returns the results.
    This function will be executed by each worker process.
    """
    cur_shadow_dir = os.path.join(privacy_dir, str(shadow_id))
    cur_shadow_dist = {}

    if model_name != "binary_diffusion":
        model = torch.load(os.path.join(cur_shadow_dir, f"{model_name}.pt"))

    for id in range(n_syn_dataset):

        if model_name != "binary_diffusion":
            # start_time = time.time()
            syn_data_pd = model.sample(len(real_data) // 2)
            # print(f"Sampling took {time.time() - start_time:.2f} seconds for shadow model {shadow_id}, dataset {id}.")
        else:
            # syn_data_pd = pd.read_csv(os.path.join(cur_shadow_dir, "sample.csv"))
            ckpt = os.path.join(cur_shadow_dir, "model-final-train.pt")
            ckpt_transformation = os.path.join(cur_shadow_dir, "transformation.joblib")
            syn_data_pd = sample_bdt(ckpt, ckpt_transformation, len(real_data) // 2) 

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
    train_shadow_models_parallel(real_data, model_params, metadata, m_shadow_models, model, privacy_dir)
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
    os.makedirs(privacy_path, exist_ok=True)

    privacy_result = evaluate_privacy(df, model_params, args.m_shadow_models, args.n_syn_dataset, model, metadata, privacy_path)

    results_path = os.path.join(privacy_path, f'privacy_result.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    dump_json(privacy_result, results_path)


if __name__ == "__main__":
    main()