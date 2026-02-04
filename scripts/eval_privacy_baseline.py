import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os

import numpy as np
import pandas as pd

from tqdm.auto import tqdm

from lib.eval_helper import normalize_data, nearest_neighbors
from lib.preprocess import clean
from lib.utils import load_config, dump_json

DATASETS_PATH = os.getenv("DATASETS_PATH", "/opt/ml/input/data/datasets")
EXP_PATH = os.getenv("EXP_PATH", "/opt/ml/output/data/exp")
N_WORKERS = int(os.getenv("N_WORKERS", 2))


def compute_baseline_self(syn_data_range, real_data, cat_features):

    cur_shadow_dist = {}

    for id in syn_data_range:

        syn_data_pd = real_data.sample(len(real_data) // 2, replace=False, random_state=id)

        raw_data_arr, syn_data_arr, n_features = normalize_data(real_data, syn_data_pd, cat_features)

        distances = nearest_neighbors(syn_data_arr, raw_data_arr)

        for i, dist in enumerate(distances):
            normalized_dist = dist[0] / np.sqrt(n_features)
            if i not in cur_shadow_dist:
                cur_shadow_dist[i] = [normalized_dist]
            else:
                cur_shadow_dist[i].append(normalized_dist)

    # get the expected distance for each record
    dists = {}
    for id, dist_list in cur_shadow_dist.items():
        mean_dist = np.mean(dist_list)
        dists[id] = [mean_dist]

    return dists


def eval_privacy_baseline_parallel(real_data, cat_features, n_syn_dataset):

    # init the distance dict
    in_member_nbrs_dist = {}
    for id in range(len(real_data)):
        in_member_nbrs_dist[id] = []

    syn_data_ranges = []
    range_length = n_syn_dataset // N_WORKERS
    for i in range(N_WORKERS):
        start = i * range_length
        end = (i + 1) * range_length if i < N_WORKERS - 1 else n_syn_dataset
        syn_data_ranges.append(range(start, end))

    baseline_args = partial(
        compute_baseline_self,
        real_data=real_data,
        cat_features=cat_features
    )

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        results = list(tqdm(
            executor.map(baseline_args, syn_data_ranges),
            total=len(syn_data_ranges),
        ))

    for dists in results:
        for id, dist_list in dists.items():
            in_member_nbrs_dist[id].extend(dist_list)

    # get the DS for each record
    DS = {}
    for id in range(len(real_data)):
        mean_in_dist = np.mean(in_member_nbrs_dist[id])
        DS[id] = abs(mean_in_dist)

    # get the MDS
    MDS = max(DS.values())

    print(f"self membership disclosure score: {MDS}")

    return {"MDS": MDS}
    

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='The name of the dataset (without extension).')
    parser.add_argument("--n_syn_dataset", "-n_syn", type=int, default=1000)

    args = parser.parse_args()

    dataset = args.dataset
    n_syn_dataset = args.n_syn_dataset

    dataset_path = os.path.join(DATASETS_PATH, f"{dataset}/{dataset}.csv")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}. Please provide a valid dataset path.")
    
    real_data = pd.read_csv(dataset_path)

    clean(real_data)

    metadata_path = os.path.join(DATASETS_PATH, dataset, 'metadata.toml')
    metadata = load_config(metadata_path)
    cat_features = metadata['cat_features']
    if metadata['task'] == 'classification':
        cat_features.append(metadata['target_column'])


    privacy_baseline = eval_privacy_baseline_parallel(real_data, cat_features, n_syn_dataset)

    results_path = os.path.join(EXP_PATH, dataset, f'privacy_result.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    dump_json(privacy_baseline, results_path)


if __name__ == "__main__":
    main()