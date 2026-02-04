import json
import os 
from pathlib import Path
import pandas as pd
import tomli
import tomli_w
from typing import Any, Union

# from lib.info import DATASETS_PATH

DATASETS_PATH = os.getenv("DATASETS_PATH", "/opt/ml/input/data/datasets")


def load_config(path: Union[Path, str]) -> Any:
    """
    load config file `.toml`
    """
    with open(path, "rb") as f:
        return tomli.load(f)


def dump_config(config, path):
    """
    dump config file `.toml`
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        tomli_w.dump(config, f)


def load_json(path: Union[Path, str]) -> Any:
    with open(path, "r") as f:
        return json.load(f)
    
    
def dump_json(config, path: Union[Path, str]):
    with open(path, "w") as f:
        json.dump(config, f, indent=4)  # Use indent for pretty printing


def load_dataset(dataset: str):
    train_path = os.path.join(DATASETS_PATH, f"{dataset}/train.csv")
    val_path = os.path.join(DATASETS_PATH, f"{dataset}/val.csv")
    test_path = os.path.join(DATASETS_PATH, f"{dataset}/test.csv")

    conditions = [
        os.path.exists(train_path),
        os.path.exists(val_path),
        os.path.exists(test_path)
    ]

    if not all(conditions):
        raise FileNotFoundError(f"Dataset {dataset} not found in {DATASETS_PATH}. Please check the path, or remember to run split.py to split the dataset into train, val and test sets.")

    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)

    return train, val, test