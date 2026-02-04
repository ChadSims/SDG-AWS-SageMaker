import os

import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR, SVC

from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

import wandb

N_TRIALS_EVALUATORS = int(os.getenv("N_TRIALS_EVALUATORS", 50))
STORAGE = os.getenv("STORAGE", "sqlite:////opt/ml/output/data/optuna_study.db")
RUN_LOCAL = os.getenv("RUN_LOCAL", "false").lower() == "true"


def train_ridge(params, X_train, y_train, X_test, y_test):
    model = Ridge(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    result = {'mse': mse, 'r2': r2}

    return result


def tune_ridge(X_train, y_train, X_val, y_val, study_name, random_state=42):

    def ridge_objective(trial):
        alpha = trial.suggest_float("alpha", 0, 10)
        fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])
        params = {"alpha": alpha, "fit_intercept": fit_intercept, "random_state": random_state}
        res = train_ridge(params, X_train, y_train, X_val, y_val)

        return res['mse']

    wandb_project_name = f"local-sdg-evaluators-{study_name}" if RUN_LOCAL else f"sagemaker-sdg-evaluators-{study_name}"
    wandb_kwargs = {"project": wandb_project_name}
    wandbc = WeightsAndBiasesCallback(
        metric_name='mse',
        wandb_kwargs=wandb_kwargs,
        as_multirun=True
    )

    try:
        optuna.delete_study(study_name=wandb_project_name, storage=STORAGE)
    except:
        pass
    
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=0),
        storage=STORAGE,
        study_name=wandb_project_name,
    )

    study.optimize(
        ridge_objective,
        n_trials=N_TRIALS_EVALUATORS,
        show_progress_bar=True,
        callbacks=[wandbc]
    )

    optim_history = optuna.visualization.plot_optimization_history(study)
    param_importance = optuna.visualization.plot_param_importances(study)

    run = wandb.init(project=wandb_project_name, name="summary_plots")
    run.log(
        {
            "optimisation_history": optim_history,
            "param_importance": param_importance
        }
    )

    wandb.finish()

    best_params = study.best_params
    best_mse = study.best_value

    return best_params, best_mse


def train_logistic(params, X_train, y_train, X_test, y_test):
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="weighted")
    accuracy = accuracy_score(y_test, y_pred)
    result = {'f1': f1, 'accuracy': accuracy}

    return result


def tune_logistic(X_train, y_train, X_val, y_val, study_name, random_state=42):

    def logistic_objective(trial):
        max_iter = trial.suggest_int("max_iter", 100, 1000)
        C = trial.suggest_float("C", 1e-5, 1e-1, log=True)
        tol = trial.suggest_float("tol", 1e-5, 1e-1, log=True)
        params = {"max_iter": max_iter, "C": C, "tol": tol, "random_state": random_state}
        res = train_logistic(params, X_train, y_train, X_val, y_val)

        return res['f1']
    
    wandb_project_name = f"local-sdg-evaluators-{study_name}" if RUN_LOCAL else f"sagemaker-sdg-evaluators-{study_name}"
    wandb_kwargs = {
        "project": wandb_project_name,
    }
    wandbc = WeightsAndBiasesCallback(
        metric_name='f1',
        wandb_kwargs=wandb_kwargs,
        as_multirun=True
    )

    try:
        optuna.delete_study(study_name=wandb_project_name, storage=STORAGE)
    except:
        pass

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=0), 
        storage=STORAGE,
        study_name=wandb_project_name,
    )

    study.optimize(
        logistic_objective,
        n_trials=N_TRIALS_EVALUATORS,
        show_progress_bar=True,
        callbacks=[wandbc]
    )

    optim_history = optuna.visualization.plot_optimization_history(study)
    param_importance = optuna.visualization.plot_param_importances(study)

    run = wandb.init(project=wandb_project_name, name="summary_plots")
    run.log(
        {
            "optimisation_history": optim_history,
            "param_importance": param_importance
        }
    )

    wandb.finish()

    best_params = study.best_params
    best_f1 = study.best_value

    return best_params, best_f1


def train_svr(params, X_train, y_train, X_test, y_test):
    model = SVR(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    result = {'mse': mse, 'r2': r2}

    return result


def tune_svr(X_train, y_train, X_val, y_val, study_name):

    def svr_objective(trial):
        C = trial.suggest_float("C", 1e-5, 1e-1, log=True)
        epsilon = trial.suggest_float("epsilon", 1e-5, 1e-1, log=True)
        params = {"C": C, "epsilon": epsilon}
        res = train_svr(params, X_train, y_train, X_val, y_val)

        return res['mse']

    wandb_project_name = f"local-sdg-evaluators-{study_name}" if RUN_LOCAL else f"sagemaker-sdg-evaluators-{study_name}"
    wandb_kwargs = {"project": wandb_project_name}
    wandbc = WeightsAndBiasesCallback(
        metric_name='mse',
        wandb_kwargs=wandb_kwargs,
        as_multirun=True
    )

    try:
        optuna.delete_study(study_name=wandb_project_name, storage=STORAGE)
    except:
        pass

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=0), 
        storage=STORAGE,
        study_name=wandb_project_name,
    )

    study.optimize(
        svr_objective,
        n_trials=N_TRIALS_EVALUATORS,
        show_progress_bar=True,
        callbacks=[wandbc]
    )

    optim_history = optuna.visualization.plot_optimization_history(study)
    param_importance = optuna.visualization.plot_param_importances(study)

    run = wandb.init(project=wandb_project_name, name="summary_plots")
    run.log(
        {
            "optimisation_history": optim_history,
            "param_importance": param_importance
        }
    )

    wandb.finish()

    best_params = study.best_params
    best_mse = study.best_value

    return best_params, best_mse


def train_svc(params, X_train, y_train, X_test, y_test):
    model = SVC(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="weighted")
    accuracy = accuracy_score(y_test, y_pred)
    result = {'f1': f1, 'accuracy': accuracy}

    return result


def tune_svc(X_train, y_train, X_val, y_val, study_name, random_state=42):

    def svc_objective(trial):
        C = trial.suggest_float("C", 1e-5, 1e-1, log=True)
        kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
        gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
        params = {"C": C, "kernel": kernel, "gamma": gamma, "random_state": random_state}
        res = train_svc(params, X_train, y_train, X_val, y_val)

        return res['f1']

    wandb_project_name = f"local-sdg-evaluators-{study_name}" if RUN_LOCAL else f"sagemaker-sdg-evaluators-{study_name}"
    wandb_kwargs = {"project": wandb_project_name,}
    wandbc = WeightsAndBiasesCallback(
        metric_name='f1',
        wandb_kwargs=wandb_kwargs,
        as_multirun=True
    )

    try:
        optuna.delete_study(study_name=wandb_project_name, storage=STORAGE)
    except:
        pass

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=0), 
        storage=STORAGE,
        study_name=wandb_project_name,
    )

    study.optimize(
        svc_objective,
        n_trials=N_TRIALS_EVALUATORS,
        show_progress_bar=True,
        callbacks=[wandbc]
    )

    optim_history = optuna.visualization.plot_optimization_history(study)
    param_importance = optuna.visualization.plot_param_importances(study)

    run = wandb.init(project=wandb_project_name, name="summary_plots")
    run.log(
        {
            "optimisation_history": optim_history,
            "param_importance": param_importance
        }
    )

    wandb.finish()

    best_params = study.best_params
    best_f1 = study.best_value

    return best_params, best_f1


def train_rfr(params, X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    result = {'mse': mse, 'r2': r2}

    return result


def tune_rfr(X_train, y_train, X_val, y_val, study_name, random_state=42):

    def rf_objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 10, 200)
        max_depth = trial.suggest_int("max_depth", 4, 64)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 8)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 8)
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "random_state": random_state
        }
        res = train_rfr(params, X_train, y_train, X_val, y_val)

        return res['mse']
    
    wandb_project_name = f"local-sdg-evaluators-{study_name}" if RUN_LOCAL else f"sagemaker-sdg-evaluators-{study_name}"
    wandb_kwargs = {"project": wandb_project_name}
    wandbc = WeightsAndBiasesCallback(
        metric_name='mse',
        wandb_kwargs=wandb_kwargs,
        as_multirun=True
    )

    try:
        optuna.delete_study(study_name=wandb_project_name, storage=STORAGE)
    except:
        pass

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=0), 
        storage=STORAGE,
        study_name=wandb_project_name,
    )

    study.optimize(
        rf_objective,
        n_trials=N_TRIALS_EVALUATORS,
        show_progress_bar=True,
        callbacks=[wandbc]
    )

    optim_history = optuna.visualization.plot_optimization_history(study)
    param_importance = optuna.visualization.plot_param_importances(study)

    run = wandb.init(project=wandb_project_name, name="summary_plots")
    run.log(
        {
            "optimisation_history": optim_history,
            "param_importance": param_importance
        }
    )

    wandb.finish()

    best_params = study.best_params
    best_mse = study.best_value

    return best_params, best_mse


def train_rfc(params, X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="weighted")
    accuracy = accuracy_score(y_test, y_pred)
    result = {'f1': f1, 'accuracy': accuracy}

    return result


def tune_rfc(X_train, y_train, X_val, y_val, study_name, random_state=42):

    def rf_objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 10, 200)
        max_depth = trial.suggest_int("max_depth", 4, 64)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 8)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 8)
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "random_state": random_state
        }
        res = train_rfc(params, X_train, y_train, X_val, y_val)

        return res['f1']
    
    wandb_project_name = f"local-sdg-evaluators-{study_name}" if RUN_LOCAL else f"sagemaker-sdg-evaluators-{study_name}"
    wandb_kwargs = {"project": wandb_project_name,}
    wandbc = WeightsAndBiasesCallback(
        metric_name='f1',
        wandb_kwargs=wandb_kwargs,
        as_multirun=True
    )

    try:
        optuna.delete_study(study_name=wandb_project_name, storage=STORAGE)
    except:
        pass

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=0), 
        storage=STORAGE,
        study_name=wandb_project_name,
    )

    study.optimize(
        rf_objective,
        n_trials=N_TRIALS_EVALUATORS,
        show_progress_bar=True,
        callbacks=[wandbc]
    )

    optim_history = optuna.visualization.plot_optimization_history(study)
    param_importance = optuna.visualization.plot_param_importances(study)

    run = wandb.init(project=wandb_project_name, name="summary_plots")
    run.log(
        {
            "optimisation_history": optim_history,
            "param_importance": param_importance
        }
    )

    wandb.finish()

    best_params = study.best_params
    best_f1 = study.best_value

    return best_params, best_f1


def train_mlpr(params, X_train, y_train, X_test, y_test):
    model = MLPRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    result = {'mse': mse, 'r2': r2}

    return result


def tune_mlpr(X_train, y_train, X_val, y_val, study_name, random_state=42):

    def mlp_objective(trial):
        max_iter = trial.suggest_int("max_iter", 50, 200)
        alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
        params = {"alpha": alpha, "max_iter": max_iter, "random_state": random_state}
        res = train_mlpr(params, X_train, y_train, X_val, y_val)

        return res['mse']
    
    wandb_project_name = f"local-sdg-evaluators-{study_name}" if RUN_LOCAL else f"sagemaker-sdg-evaluators-{study_name}"
    wandb_kwargs = {"project": wandb_project_name}
    wandbc = WeightsAndBiasesCallback(
        metric_name='mse',
        wandb_kwargs=wandb_kwargs,
        as_multirun=True
    )

    try:
        optuna.delete_study(study_name=wandb_project_name, storage=STORAGE)
    except:
        pass

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=0), 
        storage=STORAGE,
        study_name=wandb_project_name,
    )

    study.optimize(
        mlp_objective,
        n_trials=N_TRIALS_EVALUATORS,
        show_progress_bar=True,
        callbacks=[wandbc]
    )

    optim_history = optuna.visualization.plot_optimization_history(study)
    param_importance = optuna.visualization.plot_param_importances(study)

    run = wandb.init(project=wandb_project_name, name="summary_plots")
    run.log(
        {
            "optimisation_history": optim_history,
            "param_importance": param_importance
        }
    )

    wandb.finish()

    best_params = study.best_params
    best_mse = study.best_value

    return best_params, best_mse


def train_mlpc(params, X_train, y_train, X_test, y_test):
    model = MLPClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="weighted")
    accuracy = accuracy_score(y_test, y_pred)
    result = {'f1': f1, 'accuracy': accuracy}

    return result


def tune_mlpc(X_train, y_train, X_val, y_val, study_name, random_state=42):

    def mlp_objective(trial):
        max_iter = trial.suggest_int("max_iter", 50, 200)
        alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
        params = {"alpha": alpha, "max_iter": max_iter, "random_state": random_state}
        res = train_mlpc(params, X_train, y_train, X_val, y_val)

        return res['f1']
    
    wandb_project_name = f"local-sdg-evaluators-{study_name}" if RUN_LOCAL else f"sagemaker-sdg-evaluators-{study_name}"
    wandb_kwargs = {"project": wandb_project_name,}
    wandbc = WeightsAndBiasesCallback(
        metric_name='f1',
        wandb_kwargs=wandb_kwargs,
        as_multirun=True
    )

    try:
        optuna.delete_study(study_name=wandb_project_name, storage=STORAGE)
    except:
        pass

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=0), 
        storage=STORAGE,
        study_name=wandb_project_name,
    )

    study.optimize(
        mlp_objective,
        n_trials=N_TRIALS_EVALUATORS,
        show_progress_bar=True,
        callbacks=[wandbc]
    )

    optim_history = optuna.visualization.plot_optimization_history(study)
    param_importance = optuna.visualization.plot_param_importances(study)

    run = wandb.init(project=wandb_project_name, name="summary_plots")
    run.log(
        {
            "optimisation_history": optim_history,
            "param_importance": param_importance
        }
    )

    wandb.finish()

    best_params = study.best_params
    best_f1 = study.best_value

    return best_params, best_f1