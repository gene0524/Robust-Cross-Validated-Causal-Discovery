"""
rcv_framework.py

This module implements a generic Robust Cross-Validation framework for causal discovery methods.
The framework can be used with different base methods (like VAR-LiNGAM, PCMCI, etc.) to improve
the reliability of causal discovery in time series data.

Includes both the base RCV implementation and grid search functionality.

Author: Gene Yu
Date: August 2024
"""

import numpy as np
from sklearn.model_selection import KFold
from itertools import product
from src.causal_matrix_evaluation import evaluate_causal_matrices

def run_rcv(data, base_method, n_splits=5, consistency_threshold=0.4, 
            variability_threshold=0.4, adjustment_weight=0, **method_params):
    """
    Generic RCV implementation that can be used with different base methods.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input time series data
    base_method : function
        Base causal discovery method to use (e.g., run_varlingam, run_pcmci)
    n_splits : int
        Number of splits for cross-validation
    consistency_threshold : float
        Threshold for consistency check
    variability_threshold : float
        Threshold for variability check
    adjustment_weight : float
        Weight for adjusting the initial estimates
    method_params : dict
        Additional parameters specific to the base method
        
    Returns:
    --------
    list
        List of validated adjacency matrices
    """
    # Initial fit with all data
    initial_matrices = base_method(data, **method_params)
    n_lags = len(initial_matrices)
    n_vars = initial_matrices[0].shape[0]

    # Cross-validation
    kf = KFold(n_splits=n_splits)
    all_matrices = []
    
    for train_index, _ in kf.split(data):
        train_data = data[train_index]
        cv_matrices = base_method(train_data, **method_params)
        padded_matrices = pad_or_truncate_matrices(cv_matrices, n_lags-1, n_vars)
        all_matrices.append(padded_matrices)
    
    # Validation and adjustment
    validated_matrices = validate_and_adjust_matrices(
        initial_matrices, all_matrices, 
        consistency_threshold, variability_threshold, 
        adjustment_weight
    )
    
    return validated_matrices

def pad_or_truncate_matrices(matrices, target_n_lags, n_vars):
    """Pad or truncate matrices to match target number of lags"""
    current_n_lags = len(matrices) - 1
    
    if current_n_lags < target_n_lags:
        padding = [np.zeros((n_vars, n_vars)) for _ in range(target_n_lags - current_n_lags)]
        matrices = list(matrices) + padding
    elif current_n_lags > target_n_lags:
        matrices = matrices[:target_n_lags + 1]
    
    return matrices

def remove_outliers(data):
    """Remove outliers using IQR method"""
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return [x for x in data if lower_bound <= x <= upper_bound]

def validate_and_adjust_matrices(initial_matrices, all_matrices, consistency_threshold, 
                               variability_threshold, adjustment_weight):
    """Validate and adjust matrices based on cross-validation results"""
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
                    cleaned_values = remove_outliers(fold_values)
                    mean_value = np.mean(cleaned_values) if cleaned_values else initial_value
                    adjusted_value = (1 - adjustment_weight) * initial_value + adjustment_weight * mean_value
                    validated_matrix[i, j] = adjusted_value
                
        validated_matrices.append(validated_matrix)
    
    return validated_matrices

def grid_search_rcv(data, true_matrices, base_method, param_grid=None, method_params=None):
    """
    Perform grid search for RCV parameters with any base causal discovery method.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input time series data
    true_matrices : list
        True adjacency matrices for evaluation
    base_method : function
        Base causal discovery method (e.g., run_varlingam, run_pcmci)
    param_grid : dict, optional
        Grid of RCV parameters to search. If None, uses default grid
    method_params : dict, optional
        Additional parameters for the base method
        
    Returns:
    --------
    dict
        Contains best parameters, best score, and best matrices
    """
    if param_grid is None:
        param_grid = {
            'n_splits': [3, 5, 7],
            'consistency_threshold': [0.3, 0.5, 0.7],
            'variability_threshold': [0.3, 0.5, 0.7],
            'adjustment_weight': [0.0, 0.1, 0.2]
        }
    
    if method_params is None:
        method_params = {}

    # Generate all parameter combinations
    param_combinations = list(product(*param_grid.values()))
    param_keys = list(param_grid.keys())
    
    best_score = float('-inf')
    best_params = None
    best_matrices = None
    
    # Try each parameter combination
    for params in param_combinations:
        current_params = dict(zip(param_keys, params))
        
        # Run RCV with current parameters
        matrices = run_rcv(
            data=data,
            base_method=base_method,
            **current_params,
            **method_params
        )
        
        # Evaluate results
        scores = evaluate_causal_matrices(true_matrices, matrices)
        current_score = scores['f1']
        
        # Update best results if current score is better
        if current_score > best_score:
            best_score = current_score
            best_params = current_params
            best_matrices = matrices
            
    return {
        'best_params': best_params,
        'best_score': best_score,
        'best_matrices': best_matrices
    }