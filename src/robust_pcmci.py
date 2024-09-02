"""
robust_pcmci.py

This module implements a Robust Cross-Validated Peter and Clark Momentary Conditional Independence method (RCV-PCMCI).
It extends the traditional PCMCI by incorporating cross-validation and variability checks to improve the reliability
of causal discovery in time series data.

The main function, run_rcv_pcmci, performs the following steps:
1. Fits an initial PCMCI model on the entire dataset.
2. Performs k-fold cross-validation to assess the variability of causal relationships.
3. Validates and adjusts the initial model based on consistency and variability criteria.
4. Returns a set of adjacency matrices representing robust causal relationships.

This method is particularly useful for identifying stable causal structures in time series data,
while mitigating the effects of noise and statistical fluctuations.

Author: Gene Yu
Date: August 2024
"""


import numpy as np
from sklearn.model_selection import KFold
from src.run_causal_discovery import run_pcmci
from src.causal_matrix_evaluation import evaluate_causal_matrices
from itertools import product

def run_rcv_pcmci(data, n_splits=7, consistency_threshold=0.7, variability_threshold=0.4, adjustment_weight=0):
    # Initial fit with all data
    initial_matrices = run_pcmci(data)
    n_lags = len(initial_matrices) - 1
    n_vars = initial_matrices[0].shape[0]

    kf = KFold(n_splits=n_splits)
    all_adjacency_matrices = []
    
    for train_index, _ in kf.split(data):
        train_data = data[train_index]
        fit_results_matrices = run_pcmci(train_data, tau_max=n_lags)
        # Pad or truncate the result to match the initial number of lags
        padded_matrices = pad_or_truncate_matrices(fit_results_matrices, n_lags, n_vars)
        all_adjacency_matrices.append(padded_matrices)
    
    # Validation and adjustment process
    validated_matrices = validate_and_adjust_matrices(initial_matrices, all_adjacency_matrices, 
                                                      consistency_threshold, variability_threshold, adjustment_weight)
    
    return validated_matrices


def pad_or_truncate_matrices(matrices, target_n_lags, n_vars):
    """
    Pad or truncate the matrices to match the target number of lags.

    Parameters:
    matrices (list): List of adjacency matrices.
    target_n_lags (int): Target number of lags.
    n_vars (int): Number of variables.

    Returns:
    list: Padded or truncated list of adjacency matrices.
    """
    current_n_lags = len(matrices) - 1    
    
    if current_n_lags < target_n_lags:
        # Pad with zero matrices
        padding = np.zeros((target_n_lags - current_n_lags, n_vars, n_vars))
        padded_matrices = np.vstack((matrices, padding))
    elif current_n_lags > target_n_lags:
        # Truncate
        padded_matrices = matrices[:target_n_lags]
    else:
        # No change needed
        padded_matrices = matrices
    
    return padded_matrices


def remove_outliers(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return [x for x in data if lower_bound <= x <= upper_bound]

def validate_and_adjust_matrices(initial_matrices, all_matrices, consistency_threshold, variability_threshold, adjustment_weight):
    n_lags = len(initial_matrices)
    n_vars = initial_matrices[0].shape[0]
    
    validated_matrices = []
    for lag in range(n_lags):
        validated_matrix = np.zeros((n_vars, n_vars))
        for i in range(n_vars):
            for j in range(n_vars):
                initial_value = initial_matrices[lag][i, j]
                fold_values = [m[lag][i, j] for m in all_matrices]
                
                # Check consistency
                consistent_count = sum(1 for v in fold_values if np.sign(v) == np.sign(initial_value))
                consistency = consistent_count / len(fold_values)
                
                # Check variability
                variability = np.std(fold_values) / (np.abs(initial_value) + 1e-8)
                
                if consistency > consistency_threshold and variability < variability_threshold:
                    # Remove outliers and calculate mean
                    cleaned_fold_values = remove_outliers(fold_values)
                    if cleaned_fold_values:  # Check if there are any values left after removing outliers
                        mean_fold_value = np.mean(cleaned_fold_values)
                    else:
                        mean_fold_value = initial_value  # If all values are outliers, use the initial value
                    
                    # Calculate adjusted value
                    adjusted_value = (1 - adjustment_weight) * initial_value + adjustment_weight * mean_fold_value
                    validated_matrix[i, j] = adjusted_value
                else:
                    validated_matrix[i, j] = 0  # Remove unstable or inconsistent relations
        
        validated_matrices.append(validated_matrix)
    
    return validated_matrices


def grid_search_rcv_pcmci(data, true_matrices, param_grid=None):
    """
    Perform grid search to find the best parameters for RCV-PCMCI.

    Args:
    data (np.array): The input time series data.
    true_matrices (list): List of true causal matrices.
    param_grid (dict, optional): Dictionary with parameters names (str) as keys and lists of parameter settings to try as values.

    Returns:
    dict: Best parameters, best score, and best matrices.
    """
    if param_grid is None:
        param_grid = {
            'n_splits': range(3, 8, 2),
            'consistency_threshold': np.arange(0.1, 1.0, 0.3),
            'variability_threshold': np.arange(0.1, 1.0, 0.3),
            'adjustment_weight': np.arange(0, 0.3, 0.1)
        }

    param_combinations = list(product(*param_grid.values()))
    best_score = 0
    best_params = None
    best_matrices = None

    for params in param_combinations:
        current_params = dict(zip(param_grid.keys(), params))
        
        validated_matrices = run_rcv_pcmci(data, **current_params)
        
        evaluation_results = evaluate_causal_matrices(true_matrices, validated_matrices)
        current_score = evaluation_results['f1_directed']  # Using F1 score (directed) as the score

        if current_score > best_score:
            best_score = current_score
            best_params = current_params
            best_matrices = validated_matrices

    return {
        'best_params': best_params,
        'best_score': best_score,
        'best_matrices': best_matrices
    }
