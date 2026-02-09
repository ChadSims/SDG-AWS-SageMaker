import argparse
import logging
import os 
import sys
import pandas as pd

from lib.evaluation import cal_fidelity
from lib.preprocess import clean
from lib.utils import dump_json

DATASETS_PATH = os.getenv("DATASETS_PATH", "/opt/ml/input/data/datasets")
EXP_PATH = os.getenv("EXP_PATH", "/opt/ml/output/data/exp")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - [Proc:%(process)d] - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Name of the real dataset to evaluate.')
    parser.add_argument('--model', type=str, required=True, help='Name of the model used to generate the synthetic dataset.')

    args = parser.parse_args()

    dataset = args.dataset
    model = args.model

    dataset_path = os.path.join(DATASETS_PATH, f"{dataset}/{dataset}.csv")
    synthetic_dataset_path = os.path.join(EXP_PATH, f'{dataset}/{model}/sample.csv')

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}. Please provide a valid dataset path.")

    if not os.path.exists(synthetic_dataset_path):
        logger.info(f"Synthetic dataset not found at {synthetic_dataset_path}.")
        
        # Attempt to find the synthetic dataset in the input directory
        synthetic_dataset_path = synthetic_dataset_path.replace('output', 'input')

        logger.info(f"Attempting to load synthetic dataset from {synthetic_dataset_path}")

        if not os.path.exists(synthetic_dataset_path):
            raise FileNotFoundError(f"Synthetic dataset file not found at {synthetic_dataset_path}. Please generate the synthetic dataset first or move it to input/data/exp/dataset/model/sample.csv.")

    real_data = pd.read_csv(dataset_path)

    synthetic_data = pd.read_csv(synthetic_dataset_path)

    if real_data.shape[1] != synthetic_data.shape[1]:
        raise ValueError("Real and synthetic datasets must have the same number of columns.")
    # if real_data.columns.tolist() != synthetic_data.columns.tolist():
    #     raise ValueError("Real and synthetic datasets must have the same column names.")

    clean(real_data)
    clean(synthetic_data)
    
    fidelity_score = cal_fidelity(real_data, synthetic_data[real_data.columns])

    results_path = os.path.join(EXP_PATH, f'{dataset}/{model}/fidelity_result.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    dump_json({'wasserstein_gower': fidelity_score}, results_path)


if __name__ == "__main__":
    main()
