import argparse
import logging
import os
import sys

import pandas as pd
import wandb

from lib.eval_helper import evaluate_privacy
from lib.preprocess import clean
from lib.utils import dump_json, load_config

DATASETS_PATH = os.getenv("DATASETS_PATH", "/opt/ml/input/data/datasets")
PARAMS_PATH = os.getenv("PARAMS_PATH", "/opt/ml/output/data/params")
EXP_PATH = os.getenv("EXP_PATH", "/opt/ml/output/data/exp")
M_SHADOW_MODELS = int(os.getenv("M_SHADOW_MODELS", 20))
N_SYN_DATASET = int(os.getenv("N_SYN_DATASET", 100))
RUN_LOCAL = os.getenv("RUN_LOCAL", "false").lower() == "true"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - [Proc:%(process)d] - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

logger = logging.getLogger(__name__)

wandb.login()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='The name of the dataset (without extension).')
    parser.add_argument('--model', type=str, help='Name of the model (synthesiser) to evaluate privacy for.')

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

    privacy_result = evaluate_privacy(df, model_params, M_SHADOW_MODELS, N_SYN_DATASET, model, metadata, privacy_path)

    results_path = os.path.join(privacy_path, f'privacy_result.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    dump_json(privacy_result, results_path)


if __name__ == "__main__":
    main()