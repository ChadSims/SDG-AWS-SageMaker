import numpy as np 
import pandas as pd 

import gower
from scipy.stats import chi2_contingency
from sklearn.metrics.pairwise import rbf_kernel
import ot
from tqdm.auto import tqdm


#****************************************************************************************************************
# Metrics that only work for numerical data
#****************************************************************************************************************


def mmd_rbf(X, Y, gamma=1.0):
    """
    author: Jindong Wang [https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_numpy_sklearn.py]
    
    MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))

    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})

    Returns:
        [scalar] -- [MMD value]
    """
    K_XX = rbf_kernel(X, X, gamma)
    K_XY = rbf_kernel(X, Y, gamma)
    K_YY = rbf_kernel(Y, Y, gamma)
    return np.mean(K_XX) - 2 * np.mean(K_XY) + np.mean(K_YY)

def avg_change_in_corr(X:pd.DataFrame, Y:pd.DataFrame) -> float:
    """
    Compute the average change in correlation between X and Y
    Lower is better
    """
    assert X.shape[1] == Y.shape[1], "X and Y must have the same number of columns"
    return (X.corr() - Y.corr()).abs().mean().mean()

def mean_gower(X:pd.DataFrame, Y:pd.DataFrame) -> float:
    """
    Compute the mean Gower distance between X and Y
    """
    assert X.shape[1] == Y.shape[1], "X and Y must have the same number of columns"
    return np.mean(gower.gower_matrix(X, Y))

def W1(X:pd.DataFrame, Y:pd.DataFrame) -> float:
    """
    Compute the Wasserstein distance between X and Y
    Lower is better 
    """
    assert X.shape[1] == Y.shape[1], "X and Y must have the same number of columns"
    
    if isinstance(X, pd.DataFrame) and isinstance(Y, pd.DataFrame):
        M = ot.dist(X.to_numpy(), Y.to_numpy())
    elif isinstance(X, np.ndarray) and isinstance(Y, np.ndarray):
        M = ot.dist(X, Y)

    return ot.emd2([], [], M)


#****************************************************************************************************************
# Metrics that work for numerical and categorical data
#****************************************************************************************************************


def wasserstein_gower(X: pd.DataFrame, Y: pd.DataFrame) -> float:
    """
    Compute the Wasserstein distance using Gower distance between X and Y.
    X and Y should have the same number of columns, but they can have different number of rows.
    """
    gower_dist = gower.gower_matrix(X, Y)

    return ot.emd2([], [], gower_dist)

def cal_fidelity(real_data: pd.DataFrame, synthetic_data: pd.DataFrame, tune=False):

   
    if tune and real_data.shape[0] > 5000:
        n = 5
        w_gower = 0
        for i in tqdm(range(n)):
            real_sample = real_data.sample(5000, replace=True, random_state=i)
            synthetic_sample = synthetic_data.sample(5000, replace=True, random_state=i)
            w_gower += wasserstein_gower(real_sample, synthetic_sample)

        w_gower /= n
    elif real_data.shape[0] > 10000:
        n = 10
        w_gower = 0
        for i in tqdm(range(n)):
            real_sample = real_data.sample(10000, replace=True, random_state=i)
            synthetic_sample = synthetic_data.sample(10000, replace=True, random_state=i)
            w_gower += wasserstein_gower(real_sample, synthetic_sample)

        w_gower /= n
    else:
         w_gower = wasserstein_gower(real_data, synthetic_data)

    return w_gower

def chi2(x, y):
    """
    Compute the Chi-squared statistic between two categorical variables.
    'An often quoted guideline for the validity of this calculation is 
    that the test should be used only if the observed and expected frequencies in each cell are at least 5.'
    """
    contingency_table = pd.crosstab(x, y) # contingency_table.min().min() >= 5
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    return chi2, p, dof, expected

def chi2_score(X: pd.DataFrame, Y: pd.DataFrame, cat_features) -> float:
    # Ensure categorical features exist in both X and Y
    assert all(feature in X.columns for feature in cat_features), "Some categorical features are missing in X"
    assert all(feature in Y.columns for feature in cat_features), "Some categorical features are missing in Y"

    chi2_features = {}
    for col in cat_features:
        chi2_value, p_value, dof, expected = chi2(X[col], Y[col])
        chi2_features[col] = {
            'chi2': chi2_value,
            'p_value': p_value,
        }

    return chi2_features

def correlations(X: pd.DataFrame, Y: pd.DataFrame, cat_features):
    num_features = [col for col in X.columns if col not in cat_features]
    num_corr = X[num_features].corrwith(Y[num_features]).to_dict()
    cat_corr = chi2_score(X, Y, cat_features)
    return {
        'numerical': num_corr,
        'categorical': cat_corr
    }