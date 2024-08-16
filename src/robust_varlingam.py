"""
robust_varlingam.py

This module implements a Robust Cross-Validated Vector Autoregressive Linear Non-Gaussian Acyclic Model (RCV-VAR-LiNGAM).
It extends the traditional VAR-LiNGAM by incorporating cross-validation and stability checks to improve the reliability
of causal discovery in time series data.

The main function, run_rcv_varlingam, performs the following steps:
1. Fits an initial VAR-LiNGAM model on the entire dataset.
2. Performs k-fold cross-validation to assess the stability of causal relationships.
3. Validates and adjusts the initial model based on consistency and stability criteria.
4. Returns a set of adjacency matrices representing robust causal relationships.

This method is particularly useful for identifying stable causal structures in time series data,
while mitigating the effects of noise and statistical fluctuations.

Author: Gene Yu
Date: August 2024
"""

import numpy as np
from sklearn.model_selection import KFold
from src.run_causal_discovery import run_varlingam
from src.causal_matrix_evaluation import evaluate_causal_matrices
from itertools import product


def run_rcv_varlingam(data, n_splits=7, consistency_threshold=0.7, stability_threshold=0.4, adjustment_weight=0.1):
    """
    Run Robust Cross-Validated VAR-LiNGAM on the given data.

    Parameters:
    data (np.array): The input time series data.
    n_splits (int): Number of splits for k-fold cross-validation.
    consistency_threshold (float): Threshold for consistency check.
    stability_threshold (float): Threshold for stability check.
    adjustment_weight (float): Weight for adjusting the initial estimates.

    Returns:
    list: A list of adjacency matrices representing robust causal relationships.
    """
    # Initial fit with all data
    initial_fit = run_varlingam(data)
    initial_matrices = initial_fit.adjacency_matrices_
    n_lags = len(initial_matrices) - 1

    kf = KFold(n_splits=n_splits)
    all_adjacency_matrices = []
    
    for train_index, _ in kf.split(data):
        train_data = data[train_index]
        fit_results = run_varlingam(train_data, lags=n_lags)
        all_adjacency_matrices.append(fit_results.adjacency_matrices_)
    
    # Validation and adjustment process
    validated_matrices = validate_and_adjust_matrices(initial_matrices, all_adjacency_matrices, 
                                                      consistency_threshold, stability_threshold, adjustment_weight)
    
    return validated_matrices

def remove_outliers(data):
    """
    Remove outliers from the data using the Interquartile Range (IQR) method.

    Parameters:
    data (list): List of values to process.

    Returns:
    list: Data with outliers removed.
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return [x for x in data if lower_bound <= x <= upper_bound]

def validate_and_adjust_matrices(initial_matrices, all_matrices, consistency_threshold, stability_threshold, adjustment_weight):
    """
    Validate and adjust the initial matrices based on cross-validation results.

    Parameters:
    initial_matrices (list): Initial adjacency matrices from VAR-LiNGAM.
    all_matrices (list): List of adjacency matrices from cross-validation.
    consistency_threshold (float): Threshold for consistency check.
    stability_threshold (float): Threshold for stability check.
    adjustment_weight (float): Weight for adjusting the initial estimates.

    Returns:
    list: Validated and adjusted adjacency matrices.
    """
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
                
                # Check stability
                stability = np.std(fold_values) / (np.abs(initial_value) + 1e-8)
                
                if consistency > consistency_threshold and stability < stability_threshold:
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


def grid_search_rcv_varlingam(data, true_matrices, param_grid=None):
    """
    Perform grid search to find the best parameters for RCV-VARLiNGAM.

    Args:
    data (np.array): The input time series data.
    true_matrices (list): List of true causal matrices.
    param_grid (dict, optional): Dictionary with parameters names (str) as keys and lists of parameter settings to try as values.

    Returns:
    dict: Best parameters, best score, and best matrices.
    """
    if param_grid is None:
        param_grid = {
            'n_splits': range(3, 10, 2),
            'consistency_threshold': np.arange(0.1, 1.0, 0.2),
            'stability_threshold': np.arange(0.1, 1.0, 0.1),
            'adjustment_weight': np.arange(0.1, 0.5, 0.1)
        }

    param_combinations = list(product(*param_grid.values()))
    best_score = float('inf')
    best_params = None
    best_matrices = None

    for params in param_combinations:
        current_params = dict(zip(param_grid.keys(), params))
        
        validated_matrices = run_rcv_varlingam(data, **current_params)
        
        evaluation_results = evaluate_causal_matrices(true_matrices, validated_matrices)
        current_score = evaluation_results['fro']  # Using Frobenius norm as the score

        if current_score < best_score:
            best_score = current_score
            best_params = current_params
            best_matrices = validated_matrices

    return {
        'best_params': best_params,
        'best_score': best_score,
        'best_matrices': best_matrices
    }
