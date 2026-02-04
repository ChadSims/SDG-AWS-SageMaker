import argparse
from importlib import import_module
import logging
import os 
import sys

import pandas as pd
import wandb

from lib.utils import load_config, load_dataset

DATASETS_PATH = os.getenv("DATASETS_PATH", "/opt/ml/input/data/datasets")
PARAMS_PATH = os.getenv("PARAMS_PATH", "/opt/ml/output/data/params")
EXP_PATH = os.getenv("EXP_PATH", "/opt/ml/output/data/exp")
RUN_LOCAL = os.getenv("RUN_LOCAL", "false").lower() == "true"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - [Proc:%(process)d] - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

logger = logging.getLogger(__name__)

wandb.login()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='The name of the dataset (without extension).')
    parser.add_argument('--model', type=str, required=True, help='Name of the model (synthesiser) to tune.')

    args = parser.parse_args()

    dataset = args.dataset
    model = args.model

    params_path = os.path.join(PARAMS_PATH, f'synthesisers/{model}/{dataset}.toml')

    if not os.path.exists(params_path):
        logger.info(f"Parameters for synthesiser not found at {params_path}.")
    
        # Attempt to find the parameters in the input directory
        params_path = params_path.replace('output', 'input')

        logger.info(f"Attempting to load parameters from {params_path}")

        if not os.path.exists(params_path):
            raise FileNotFoundError(f"Parameters file not found at {params_path}. Please generate parameters first or move them to input/data/exp/dataset/model/synthesisers.")

    model_params = load_config(params_path)

    exp_path = os.path.join(EXP_PATH, f'{dataset}/{model}')
    os.makedirs(exp_path, exist_ok=True)

    train, val, test = load_dataset(dataset)

    train_val = pd.concat([train, val], axis=0, ignore_index=True)

    # normalise ?

    if model == 'binary_diffusion':

        metadata_path = os.path.join(DATASETS_PATH, f"{dataset}/metadata.toml")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}. Please generate metadata first.")
        metadata = load_config(metadata_path)

        project_name = f"local-sdg-synthesisers-{model}-{dataset}-train" if RUN_LOCAL else f"sagemaker-sdg-synthesisers-{model}-{dataset}-train"
        wandb.init(project=project_name, name=f'train', config=model_params)

        synthesiser = import_module(f'synthesisers.{model}')
        synthesiser.train(train_val, model_params, metadata, exp_path)

        wandb.finish()

    else:
        synthesiser = import_module(f'synthesisers.{model}')
        synthesiser.train(train_val, model_params, exp_path)


if __name__ == "__main__":
    main()