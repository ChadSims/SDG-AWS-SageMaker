import os
import pickle
import time
from importlib import import_module

import numpy as np
import pandas as pd
import torch
from sdv.metadata import Metadata
from sdv.single_table import CopulaGANSynthesizer, GaussianCopulaSynthesizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    QuantileTransformer,
    StandardScaler,
)
from tqdm import tqdm

import wandb
from lib.preprocess import clean
from synthesisers.binary_diffusion import sample as sample_bdt
from synthesisers.binary_diffusion import train as train_bdt
from synthesisers.copula_gan import init_model as copula_gan_init
from synthesisers.ctgan import init_model as ctgan_init
from synthesisers.gaussian_copula import init_model as gaussian_copula_init
from synthesisers.potnet import init_model as potnet_init
from synthesisers.potnet_core import load_model as load_potnet_model
from synthesisers.tvae import init_model as tvae_init

RUN_LOCAL = os.getenv("RUN_LOCAL", "false").lower() == "true"


def train_model(shadow_data, model_params, model_name, shadow_dir):

    metadata = Metadata.detect_from_dataframe(shadow_data)

    if model_name == "ctgan":
        model = ctgan_init(metadata, model_params)
    elif model_name == "tvae":
        model = tvae_init(metadata, model_params)
    elif model_name == "gaussian_copula":
        # model = gaussian_copula_init(metadata, model_params)
        model = GaussianCopulaSynthesizer(metadata)
    elif model_name == "copula_gan":
        # model = copula_gan_init(metadata, model_params)
        model_params.pop('numerical_distributions', None)
        model = CopulaGANSynthesizer(metadata, **model_params)
    elif model_name == "potnet":
        model = potnet_init(model_params)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.fit(shadow_data)

    # save the model
    os.makedirs(shadow_dir, exist_ok=True)
    if model_name != "potnet":
        torch.save(model, os.path.join(shadow_dir, f"{model_name}.pt"))
    else:
        model_path = os.path.join(shadow_dir, "potnet.pt")
        model.save(model_path=model_path)


def train_shadow_models(real_data, model_params, metadata, m_shadow_models, model, privacy_dir):
    """
    Train shadow models and save them. 
    """
    size = len(real_data)
    np.random.seed(0)
    keep = np.random.uniform(0, 1, size=(m_shadow_models, size))
    order = keep.argsort(0)
    keep = order < int(0.5 * m_shadow_models)

    for shadow_id in tqdm(range(m_shadow_models)):
        
        # perpare saved dir
        cur_shadow_dir = os.path.join(privacy_dir, str(shadow_id))
        os.makedirs(cur_shadow_dir, exist_ok=True)

        # membership info
        cur_keep = np.array(keep[shadow_id], dtype=bool)
        cur_member = cur_keep.nonzero()[0]

        # select data for this shadow model
        shadow_data_pd = real_data.iloc[cur_member].reset_index(drop=True)
        # print(f"{shadow_id} shadow data size: {len(shadow_data_pd)}/{len(real_data)}")

        # train model 
        if model != "binary_diffusion":
            train_model(shadow_data_pd, model_params, model, cur_shadow_dir)
        else:
            dataset = privacy_dir.split('/')[-3]
            project_name = f"local-sdg-synthesisers-{model}-{dataset}-privacy" if RUN_LOCAL else f"sagemaker-sdg-synthesisers-{model}-{dataset}-privacy"
            wandb.init(project=project_name, name=f'trial {shadow_id}', config=model_params)

            train_bdt(shadow_data_pd, model_params, metadata, cur_shadow_dir)

            wandb.finish()

        # save member info
        with open(os.path.join(cur_shadow_dir, "member.pkl"), "wb") as f:
            pickle.dump(cur_member, f)


def cat_encode(X):
    """
    one-hot encode for categorical and ordinal features
    """
    oe = OneHotEncoder(
        handle_unknown="ignore",  # type: ignore[code] handle_unknown='use_encoded_value',
        sparse_output=False,  # type: ignore[code] unknown_value=-1
    ).fit(X)

    return oe


def normalize(X, normalization="quantile"):
    """
    normalize continuous features
    """
    if normalization == "standard":
        scaler = StandardScaler()
    elif normalization == "minmax":
        scaler = MinMaxScaler()
    elif normalization == "quantile":
        # adopt from Tab-DDPM
        scaler = QuantileTransformer(
            output_distribution="normal", n_quantiles=max(min(X.shape[0] // 30, 1000), 10), subsample=int(1e8)
        )
    else:
        raise ValueError("normalization must be standard, minmax, or quantile, but got " + normalization)

    scaler.fit(X)
    return scaler


def normalize_data(raw_data_pd, syn_data_pd, discrete_cols):

    normalization = "minmax"

    raw_data_arr = []
    syn_data_arr = []

    for col in raw_data_pd.columns:
        real_data_col = raw_data_pd[col].values.reshape(-1, 1)
        syn_data_col = syn_data_pd[col].values.reshape(-1, 1)
        # set the type of synthetic data to be the same as the real data
        syn_data_col = syn_data_col.astype(real_data_col.dtype)
        # fit the scaler
        if discrete_cols and col in discrete_cols:
            scaler = cat_encode(real_data_col)
        else:
            scaler = normalize(real_data_col, normalization)

        # transform the data
        real_data_transformed = scaler.transform(real_data_col)
        syn_data_transformed = scaler.transform(syn_data_col)

        # ensure the transformed data has 2 dimensions
        if len(real_data_transformed.shape) == 1:
            real_data_transformed = real_data_transformed.reshape(-1, 1)
        if len(syn_data_transformed.shape) == 1:
            syn_data_transformed = syn_data_transformed.reshape(-1, 1)

        raw_data_arr.append(real_data_transformed)
        syn_data_arr.append(syn_data_transformed)

    # concatenate the features in the last dimension
    raw_data_arr = np.concatenate(raw_data_arr, axis=1)
    syn_data_arr = np.concatenate(syn_data_arr, axis=1)

    n_features = len(raw_data_pd.columns)

    return raw_data_arr, syn_data_arr, n_features


def nearest_neighbors(syn_data: np.array, target_data: np.array):
    """
    Find the nearest distance in syn_data for each target data.

    Parameters:
        syn_data (np.array): The synthetic data array.
        target_data (np.array): The target data array.

    Returns:
        np.array: An array of distances representing the nearest distance in syn_data for each target data.
    """
    nbrs_synth = NearestNeighbors(n_neighbors=1, n_jobs=-1, p=2).fit(syn_data)
    distances, _ = nbrs_synth.kneighbors(target_data)
    return distances


def compute_MDS(real_data, m_shadow_models, n_syn_dataset, model_name, cat_features, privacy_dir):

    # init the distance dict
    in_member_nbrs_dist = {}
    out_member_nbrs_dist = {}
    for id in range(len(real_data)):
        in_member_nbrs_dist[id] = []
        out_member_nbrs_dist[id] = []

    print("Compute distances for each record in shadow models...")
    for shadow_id in tqdm(range(m_shadow_models)):
        cur_shadow_dir = os.path.join(privacy_dir, str(shadow_id))
        cur_shadow_dist = {}

        if model_name == "potnet":
            model_path = os.path.join(cur_shadow_dir, f"{model_name}.pt")
            model = load_potnet_model(model_path)
        elif model_name != "binary_diffusion":
            model = torch.load(os.path.join(cur_shadow_dir, f"{model_name}.pt"))

        for id in range(n_syn_dataset):

            if model_name == "potnet":

                syn_data_pd = model.generate(len(real_data) // 2)

            elif model_name == "binary_diffusion":
                # syn_data_pd = pd.read_csv(os.path.join(cur_shadow_dir, "sample.csv"))
                ckpt = os.path.join(cur_shadow_dir, "model-final-train.pt")
                ckpt_transformation = os.path.join(cur_shadow_dir, "transformation.joblib")
                # start_time = time.time()
                syn_data_pd = sample_bdt(ckpt, ckpt_transformation, len(real_data) // 2)
                # print(f"Sampling took {time.time() - start_time:.2f} seconds for shadow model {shadow_id}, dataset {id}.")
            else:
                # start_time = time.time()
                syn_data_pd = model.sample(len(real_data) // 2)
                # print(f"Sampling took {time.time() - start_time:.2f} seconds for shadow model {shadow_id}, dataset {id}.")

            clean(syn_data_pd)

            raw_data_arr, syn_data_arr, n_features = normalize_data(real_data, syn_data_pd, cat_features)

            distances = nearest_neighbors(syn_data_arr, raw_data_arr)

            for i, dist in enumerate(distances):
                normalized_dist = dist[0] / np.sqrt(n_features)
                if i not in cur_shadow_dist:
                    cur_shadow_dist[i] = [normalized_dist]
                else:
                    cur_shadow_dist[i].append(normalized_dist)

        with open(os.path.join(cur_shadow_dir, "member.pkl"), "rb") as f:
            member = pickle.load(f)

        # get the expected distance for each record
        for id, dist_list in cur_shadow_dist.items():
            mean_dist = np.mean(dist_list)
            if id in member:
                in_member_nbrs_dist[id].append(mean_dist)
            else:
                out_member_nbrs_dist[id].append(mean_dist)

    # get the DS for each record
    print("Calculate disclosure score for each record...")
    DS = {}
    for id in range(len(real_data)):
        mean_in_dist = np.mean(in_member_nbrs_dist[id])
        mean_out_dist = np.mean(out_member_nbrs_dist[id])
        DS[id] = abs(mean_in_dist - mean_out_dist)
        # print(
        #     f"for record {id}, in member size: {len(in_member_nbrs_dist[id])}, out member size: {len(out_member_nbrs_dist[id])}, DS: {DS[id]}"
        # )

    # get the MDS
    MDS = max(DS.values())

    print("membership disclosure score: {}".format(MDS))

    # save the distance difference
    with open(os.path.join(privacy_dir, "disclosure_score.pkl"), "wb") as f:
        pickle.dump(DS, f)

    return MDS


def evaluate_privacy(real_data, model_params, m_shadow_models, n_syn_dataset, model, metadata, privacy_dir):

    print("Train shadow models...")
    train_shadow_models(real_data, model_params, metadata, m_shadow_models, model, privacy_dir)
    print("Shadow models trained.")

    cat_features = metadata['cat_features']
    discrete_cols = cat_features + [metadata["target_column"]] if metadata["task"] == "classification" else cat_features

    print("Compute MDS...")
    MDS = compute_MDS(real_data, m_shadow_models, n_syn_dataset, model, discrete_cols, privacy_dir)

    return {"MDS": MDS}