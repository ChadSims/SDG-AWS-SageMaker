import argparse
import os
import subprocess

ROOT_DIR = ''

PYTHON = os.path.join(ROOT_DIR, '.venv/bin/python')

# scripts
GENERATE_METADATA = os.path.join(ROOT_DIR, 'scripts/generate_metadata.py')
SPLIT = os.path.join(ROOT_DIR, 'scripts/split.py')
IMPUTE = os.path.join(ROOT_DIR, 'scripts/impute.py')
TUNE_SYNTHESISER = os.path.join(ROOT_DIR, 'scripts/tune_synthesiser.py')
TUNE_SYNTHESISER_PARALLEL = os.path.join(ROOT_DIR, 'scripts/tune_synthesiser_parallel.py')
TRAIN_SYNTHESISER = os.path.join(ROOT_DIR, 'scripts/train_synthesiser.py')
TUNE_EVALUATORS = os.path.join(ROOT_DIR, 'scripts/tune_evaluators.py')
TUNE_EVALUATORS_PARALLEL = os.path.join(ROOT_DIR, 'scripts/tune_evaluators_parallel.py')
EVAL_UTILITY = os.path.join(ROOT_DIR, 'scripts/eval_utility.py')
EVAL_FIDELITY = os.path.join(ROOT_DIR, 'scripts/eval_fidelity.py')
EVAL_PRIVACY = os.path.join(ROOT_DIR, 'scripts/eval_privacy.py')
EVAL_PRIVACY_PARALLEL = os.path.join(ROOT_DIR, 'scripts/eval_privacy_parallel.py')


parser = argparse.ArgumentParser(description="Generate metadata from a DataFrame.")
parser.add_argument('--dataset', type=str, required=True, help='The name of the dataset (without extension).')
parser.add_argument('--target_column', type=str, default=None, required=False, help='Name of the target column.')
parser.add_argument('--imputed', action='store_true', help='Whether the dataset is imputed or not.')
parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility.')

parser.add_argument('--datasets_path', type=str, default=os.path.join(ROOT_DIR, 'datasets'), help='Path to the datasets directory.')
parser.add_argument('--params_path', type=str, default=os.path.join(ROOT_DIR, 'params'), help='Path to the parameters directory.')
parser.add_argument('--exp_path', type=str, default=os.path.join(ROOT_DIR, 'exp'), help='Path to the experiment directory.')
parser.add_argument('--n_trials_evaluators', type=int, default=50, help='Number of trials for tuning evaluators.')
parser.add_argument('--n_trials_synthesisers', type=int, default=50, help='Number of trials for tuning synthesisers.')
parser.add_argument('--bd_trials', type=int, default=20, help='Number of trials for Binary Diffusion.')
parser.add_argument('--bd_train_steps', type=int, default=200000, help='Number of training steps for Binary Diffusion.')
parser.add_argument('--bd_tune_steps', type=int, default=50000, help='Number of tuning steps for Binary Diffusion.')
parser.add_argument('--storage', type=str, default='sqlite:///optuna_study.db', help='Storage path for Optuna studies.')
parser.add_argument('--journal_storage', type=str, default="journal.log", help='Journal storage for optuna studies')
parser.add_argument('--n_trials_per_worker', type=int, default=10, help='Number of trials per worker for optuna studies.')
parser.add_argument('--n_workers', type=int, default=5, help='Number of workers for optuna studies.')
parser.add_argument('--n_dataloaders', type=int, default=2, help='Number of dataloaders to use for training.')
parser.add_argument('--run_local', type=str, default='true', help='Whether to run the script locally or on SageMaker. Set to "true" for local runs.')

args = parser.parse_args()

dataset = args.dataset
if args.target_column:
    target_column = args.target_column

os.environ['DATASETS_PATH'] = args.datasets_path
os.environ['PARAMS_PATH'] = args.params_path
os.environ['EXP_PATH'] = args.exp_path
os.environ['N_TRIALS_EVALUATORS'] = str(args.n_trials_evaluators)
os.environ['N_TRIALS_SYNTHESISERS'] = str(args.n_trials_synthesisers)
os.environ['BD_TRIALS'] = str(args.bd_trials)
os.environ['BD_TRAIN_STEPS'] = str(args.bd_train_steps)
os.environ['BD_TUNE_STEPS'] = str(args.bd_tune_steps)
os.environ['STORAGE'] = args.storage
os.environ['JOURNAL_STORAGE'] = args.journal_storage
os.environ['N_TRIALS_PER_WORKER'] = str(args.n_trials_per_worker)
os.environ['N_WORKERS'] = str(args.n_workers)
os.environ['N_DATALOADERS'] = str(args.n_dataloaders)
os.environ['RUN_LOCAL'] = args.run_local.lower()

def run_script(cmd):
    try:
        result = subprocess.run(cmd, check=True)
        # print(f'return code {result.returncode}')
    except subprocess.CalledProcessError as e:
        print(e)
        print(e.returncode)

def tune_synthesiser(model):
    print(f'Tuning synthesiser for model: {model} on dataset: {dataset}')
    cmd = [
        PYTHON, TUNE_SYNTHESISER,
        '--dataset', dataset,
        '--model', model,
        '--random_state', str(args.random_state),
    ]
    run_script(cmd)

def tune_synthesiser_parallel(model):
    print(f'Tuning synthesiser in parallel for model: {model} on dataset: {dataset}')
    cmd = [
        PYTHON, TUNE_SYNTHESISER_PARALLEL,
        '--dataset', dataset,
        '--model', model,
    ]
    run_script(cmd)

def train_synthesiser(model):
    print(f'Training synthesiser for model: {model} on dataset: {dataset}')
    cmd = [
        PYTHON, TRAIN_SYNTHESISER,
        '--dataset', dataset,
        '--model', model,
    ]
    run_script(cmd)

def tune_evaluators(model):
    print(f'Tuning evaluators for model: {model} on dataset: {dataset}')
    cmd = [
        PYTHON, TUNE_EVALUATORS,
        '--dataset', dataset,
        '--model', model,
    ]
    run_script(cmd)

def tune_evaluators_parallel(model):
    print(f'Tuning evaluators in parallel for model: {model} on dataset: {dataset}')
    cmd = [
        PYTHON, TUNE_EVALUATORS_PARALLEL,
        '--dataset', dataset,
        '--model', model,
    ]
    run_script(cmd)

def eval_utility_synthetic(model):
    print(f'Evaluating utility for model: {model} on dataset: {dataset}')
    cmd = [
        PYTHON, EVAL_UTILITY,
        '--dataset', dataset,
        '--model', model,
    ]
    run_script(cmd)

def eval_fidelity(model):
    print(f'Evaluating fidelity for model: {model} on dataset: {dataset}')
    cmd = [
        PYTHON, EVAL_FIDELITY,
        '--dataset', dataset,
        '--model', model,
    ]
    run_script(cmd)

def eval_privacy(model):
    print(f'Evaluating privacy for model: {model} on dataset: {dataset}')
    cmd = [
        PYTHON, EVAL_PRIVACY,
        '--dataset', dataset,
        '--model', model,
    ]
    run_script(cmd)

def eval_privacy_parallel(model):
    print(f'Evaluating privacy in parallel for model: {model} on dataset: {dataset}')
    cmd = [
        PYTHON, EVAL_PRIVACY_PARALLEL,
        '--dataset', dataset,
        '--model', model,
    ]
    run_script(cmd)


# Tune evaluators on real data
print(f'Tuning evaluators for dataset: {dataset}')
cmd = [
    PYTHON, TUNE_EVALUATORS_PARALLEL,
    '--dataset', dataset,
]
run_script(cmd)

# Evaluate utility on real data
print(f'Evaluating utility for dataset: {dataset}')
cmd = [
    PYTHON, EVAL_UTILITY,
    '--dataset', dataset,
]
run_script(cmd)

# CTGAN 
print('Running CTGAN pipeline...')

os.environ['N_TRIALS_PER_WORKER'] = str(20)
os.environ['N_WORKERS'] = str(2)
tune_synthesiser_parallel('ctgan')
train_synthesiser('ctgan')
os.environ['N_TRIALS_PER_WORKER'] = str(10)
os.environ['N_WORKERS'] = str(5)
tune_evaluators_parallel('ctgan')
eval_utility_synthetic('ctgan')
eval_fidelity('ctgan')

eval_privacy_parallel('ctgan')
