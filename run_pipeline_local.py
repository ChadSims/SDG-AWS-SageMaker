import argparse
import os
import subprocess
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

PYTHON = os.path.join(ROOT_DIR, ".venv/bin/python")

# scripts
GENERATE_METADATA = os.path.join(ROOT_DIR, "scripts/generate_metadata.py")
SPLIT = os.path.join(ROOT_DIR, "scripts/split.py")
IMPUTE = os.path.join(ROOT_DIR, "scripts/impute.py")
TUNE_SYNTHESISER = os.path.join(ROOT_DIR, "scripts/tune_synthesiser.py")
TUNE_SYNTHESISER_PARALLEL = os.path.join(ROOT_DIR, "scripts/tune_synthesiser_parallel.py")
TRAIN_SYNTHESISER = os.path.join(ROOT_DIR, "scripts/train_synthesiser.py")
TUNE_EVALUATORS = os.path.join(ROOT_DIR, "scripts/tune_evaluators.py")
TUNE_EVALUATORS_PARALLEL = os.path.join(ROOT_DIR, "scripts/tune_evaluators_parallel.py")
EVAL_UTILITY = os.path.join(ROOT_DIR, "scripts/eval_utility.py")
EVAL_FIDELITY = os.path.join(ROOT_DIR, "scripts/eval_fidelity.py")
EVAL_PRIVACY = os.path.join(ROOT_DIR, "scripts/eval_privacy.py")
EVAL_PRIVACY_PARALLEL = os.path.join(ROOT_DIR, "scripts/eval_privacy_parallel.py")


def run_script(cmd: list) -> None:
    try:
        result = subprocess.run(cmd, check=True)
        # print(f'return code {result.returncode}')
    except subprocess.CalledProcessError as e:
        print(e)
        print(e.returncode)


def tune_synthesiser(dataset: str, model: str) -> None:
    print(f"Tuning synthesiser for model: {model} on dataset: {dataset}")
    cmd = [
        PYTHON, TUNE_SYNTHESISER,
        "--dataset", dataset,
        "--model", model,
    ]
    run_script(cmd)


def tune_synthesiser_parallel(dataset: str, model: str) -> None:
    print(f"Tuning synthesiser in parallel for model: {model} on dataset: {dataset}")
    cmd = [
        PYTHON, TUNE_SYNTHESISER_PARALLEL,
        "--dataset", dataset,
        "--model", model,
    ]
    run_script(cmd)


def train_synthesiser(dataset: str, model: str) -> None:
    print(f"Training synthesiser for model: {model} on dataset: {dataset}")
    cmd = [
        PYTHON, TRAIN_SYNTHESISER,
        "--dataset", dataset,
        "--model", model,
    ]
    run_script(cmd)


def tune_evaluators(dataset: str, model: str) -> None:
    print(f"Tuning evaluators for model: {model} on dataset: {dataset}")
    cmd = [
        PYTHON, TUNE_EVALUATORS,
        "--dataset", dataset,
        "--model", model,
    ]
    run_script(cmd)


def tune_evaluators_parallel(dataset: str, model: str) -> None:
    print(f"Tuning evaluators in parallel for model: {model} on dataset: {dataset}")
    cmd = [
        PYTHON, TUNE_EVALUATORS_PARALLEL,
        "--dataset", dataset,
        "--model", model,
    ]
    run_script(cmd)


def eval_utility_synthetic(dataset: str, model: str) -> None:
    print(f"Evaluating utility for model: {model} on dataset: {dataset}")
    cmd = [
        PYTHON, EVAL_UTILITY,
        "--dataset", dataset,
        "--model", model,
    ]
    run_script(cmd)


def eval_fidelity(dataset: str, model: str) -> None:
    print(f"Evaluating fidelity for model: {model} on dataset: {dataset}")
    cmd = [
        PYTHON, EVAL_FIDELITY,
        "--dataset", dataset,
        "--model", model,
    ]
    run_script(cmd)


def eval_privacy(dataset: str, model: str) -> None:
    print(f"Evaluating privacy for model: {model} on dataset: {dataset}")
    cmd = [
        PYTHON, EVAL_PRIVACY,
        "--dataset", dataset,
        "--model", model,
    ]
    run_script(cmd)


def eval_privacy_parallel(dataset: str, model: str) -> None:
    print(f"Evaluating privacy in parallel for model: {model} on dataset: {dataset}")
    cmd = [
        PYTHON, EVAL_PRIVACY_PARALLEL,
        "--dataset", dataset,
        "--model", model,
    ]
    run_script(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate metadata from a DataFrame.")
    # Always required
    parser.add_argument("--dataset", type=str, required=True, help="The name of the dataset (without extension).")
    parser.add_argument("--model", type=str, required=True, help='Name of the model to train: it can be "ctgan", "tvae", "copula_gan", "gaussian_copula", "binary_diffusion", or "potnet"')
    # Required for Dataset Setup only
    parser.add_argument("--target_column", type=str, default=None, required=False, help="Name of the target column.")
    parser.add_argument("--imputed", action="store_true", help="Whether the dataset is imputed or not.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of the dataset to include in the test split.")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility.")
    # Path variables
    parser.add_argument("--datasets_path", type=str, default=os.path.join(ROOT_DIR, "datasets"), help="Path to the datasets directory.")
    parser.add_argument("--params_path", type=str, default=os.path.join(ROOT_DIR, "params"), help="Path to the parameters directory.")
    parser.add_argument("--exp_path", type=str, default=os.path.join(ROOT_DIR, "exp"), help="Path to the experiment directory.")
    # Set sequential execution
    parser.add_argument("--n_trials_evaluators", type=int, default=50, help="Number of trials for tuning evaluators.")
    parser.add_argument("--n_trials_synthesisers", type=int, default=50, help="Number of trials for tuning synthesisers.")
    parser.add_argument("--bd_trials", type=int, default=20, help="Number of trials for Binary Diffusion.")
    # Binary diffusion specific params
    parser.add_argument("--bd_train_steps", type=int, default=200000, help="Number of training steps for Binary Diffusion.")
    parser.add_argument("--bd_tune_steps", type=int, default=50000, help="Number of tuning steps for Binary Diffusion.")
    parser.add_argument("--n_dataloaders", type=int, default=2, help="Number of dataloaders to use for training.")
    # Optuna storage location
    parser.add_argument("--storage", type=str, default="sqlite:///optuna_study.db", help="Storage path for Optuna studies (sequential).")
    parser.add_argument("--journal_storage", type=str, default="journal.log", help="Journal storage for optuna studies (parallel).")
    # Set parallel execution
    parser.add_argument("--n_trials_per_worker", type=int, default=10, help="Number of trials per worker for optuna studies.")
    parser.add_argument("--n_workers", type=int, default=5, help="Number of workers for optuna studies.")
    # Privacy evaluation
    parser.add_argument("--m_shadow_models", "-m_model", type=int, default=20)
    parser.add_argument("--n_syn_dataset", "-n_syn", type=int, default=100)
    # Running locally flag - true if using this script
    parser.add_argument("--run_local", type=str, default="true", help='Whether to run the script locally or on SageMaker. Set to "true" for local runs.')

    args = parser.parse_args()

    dataset = args.dataset
    model = args.model
    if args.target_column:
        target_column = args.target_column

    os.environ["DATASETS_PATH"] = args.datasets_path
    os.environ["PARAMS_PATH"] = args.params_path
    os.environ["EXP_PATH"] = args.exp_path
    os.environ["N_TRIALS_EVALUATORS"] = str(args.n_trials_evaluators)
    os.environ["N_TRIALS_SYNTHESISERS"] = str(args.n_trials_synthesisers)
    os.environ["BD_TRIALS"] = str(args.bd_trials)
    os.environ["BD_TRAIN_STEPS"] = str(args.bd_train_steps)
    os.environ["BD_TUNE_STEPS"] = str(args.bd_tune_steps)
    os.environ["N_DATALOADERS"] = str(args.n_dataloaders)
    os.environ["STORAGE"] = args.storage
    os.environ["JOURNAL_STORAGE"] = args.journal_storage
    os.environ["N_TRIALS_PER_WORKER"] = str(args.n_trials_per_worker)
    os.environ["N_WORKERS"] = str(args.n_workers)
    os.environ["M_SHADOW_MODELS"] = str(args.m_shadow_models)
    os.environ["N_SYN_DATASET"] = str(args.n_syn_dataset)
    os.environ["RUN_LOCAL"] = args.run_local.lower()

    # ********************************************
    # Dataset Setup

    # 1. Generate Metadata for dataset
    print(f'Generate Metadata for dataset: {dataset}')
    cmd = [
        PYTHON, GENERATE_METADATA,
        '--dataset', dataset,
        '--target_column', target_column,
    ]
    run_script(cmd)

    # 2. Split Dataset for training and validation
    print(f'Split dataset: {dataset}')
    cmd = [
        PYTHON, SPLIT,
        '--dataset', dataset,
        '--test_size', str(args.test_size), 
        '--random_state', str(args.random_state)
    ]
    run_script(cmd)

    # # 2.5 Impute

    # ********************************************
    # Evaluate on Real

    # 3. Tune evaluators on real data (generates ideal ML model params - stored in params/evaluators/model)
    print(f'Tuning evaluators for dataset: {dataset}')
    cmd = [
        PYTHON, TUNE_EVALUATORS_PARALLEL,
        '--dataset', dataset,
    ]
    run_script(cmd)

    # 4. Evaluate utility on real data. Result saved in exp/dataset
    print(f'Evaluating utility for dataset: {dataset}')
    cmd = [
        PYTHON, EVAL_UTILITY,
        '--dataset', dataset,
    ]
    run_script(cmd)

    # 4.5 Evaluate baseline privacy 

    # ********************************************
    # Train and evaluate generative model
    # print(f'Training {args.model} on dataset: {dataset}')

    # 5. Tune generative model (generates ideal model params - stored in params/synthesisers/model) 
    os.environ['N_TRIALS_PER_WORKER'] = str(20)
    os.environ['N_WORKERS'] = str(2)
    tune_synthesiser_parallel(dataset, model)

    # 6. Train the generative model (using ideal params - samples dataset and saves model file to exp/dataset/model)
    train_synthesiser(dataset, model)

    # 7. Tune evaluators on synthetic data (generates ideal ML model params - stored in exp/dataset/model/evaluators/model)
    os.environ['N_TRIALS_PER_WORKER'] = str(10)
    os.environ['N_WORKERS'] = str(5)
    tune_evaluators_parallel(dataset, model)

    # 8. Evaluate Utility on synthetic data. Result saved in exp/dataset/model
    eval_utility_synthetic(dataset, model)

    # 9. Evaluate Fidelity between real and synthetic data. Result saved in exp/dataset/model
    eval_fidelity(dataset, model)

    # 10. Evaluate Privacy of Synthesiser
    eval_privacy_parallel(dataset, model)
