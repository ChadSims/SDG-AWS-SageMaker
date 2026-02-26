import argparse
import os
from importlib import import_module

import pandas as pd

import wandb
from lib.utils import dump_config, load_config, load_dataset

DATASETS_PATH = os.getenv("DATASETS_PATH", "/opt/ml/input/data/datasets")
PARAMS_PATH = os.getenv("PARAMS_PATH", "/opt/ml/output/data/params")
EXP_PATH = os.getenv("EXP_PATH", "/opt/ml/output/data/exp")


wandb.login()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="The name of the dataset (without extension).")
    parser.add_argument("--model", type=str, required=True, help="Name of the model (synthesiser) to tune.")

    args = parser.parse_args()

    dataset = args.dataset
    model = args.model

    metadata_path = os.path.join(DATASETS_PATH, f"{dataset}/metadata.toml")
    metadata = load_config(metadata_path)

    random_state = metadata["random_state"]

    exp_path = os.path.join(EXP_PATH, f"{dataset}/{model}")
    os.makedirs(exp_path, exist_ok=True)

    train, val, test = load_dataset(dataset)
    train_val = pd.concat([train, val], axis=0, ignore_index=True)

    study_name = f"{model}-{dataset}"

    if model == "binary_diffusion":
        synthesiser = import_module(f"synthesisers.{model}")

        best_params = synthesiser.tune(
            train_val, metadata, exp_path, study_name, seed=random_state
        )

    elif model == "potnet":
        synthesiser = import_module(f"synthesisers.{model}")
        best_params = synthesiser.tune(train_val, metadata, study_name)

    else:
        synthesiser = import_module(f"synthesisers.{model}")
        best_params = synthesiser.tune(train, val, study_name)

    # save parameters
    params_path = os.path.join(PARAMS_PATH, f"synthesisers/{model}/{dataset}.toml")
    os.makedirs(os.path.dirname(params_path), exist_ok=True)

    dump_config(best_params, params_path)


if __name__ == "__main__":
    main()
