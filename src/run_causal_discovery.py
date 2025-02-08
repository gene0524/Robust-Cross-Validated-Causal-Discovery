import numpy as np
import pandas as pd
import os
from src.models.lingam_master import lingam
from src.models.tigramite_master.tigramite.pcmci import PCMCI
from src.models.tigramite_master.tigramite.independence_tests.parcorr import ParCorr
from src.models.tigramite_master.tigramite import data_processing as pp
from src.models.TCDF_master import TCDF
from src.causal_matrix_evaluation import evaluate_causal_matrices
from src.rcd_framework import run_rcd, grid_search_rcd
from itertools import product

def run_varlingam(data, lags=3):
    """
    Run VAR-LiNGAM on the given data.
    
    :param data: pandas DataFrame
    :param lags: number of lags to include (default: 5)
    :return: VARLiNGAM results
    """
    model = lingam.VARLiNGAM(lags=lags, prune=True)
    results = model.fit(data)
    return results.adjacency_matrices_


def run_pcmci(data, columns=None, alpha=0.05, tau_max=3):
    """
    Run PCMCI on the given data.
    
    :param data: pandas DataFrame
    :param alpha: significance level (default: 0.05)
    :param tau_max: maximum time lag to test (default: 5)
    :return: PCMCI results
    """
    dataframe = pp.DataFrame(data, 
                             datatime=np.arange(len(data)), 
                             var_names=columns)
    parcorr = ParCorr(significance='analytic')
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=1)
    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=alpha)
    adj_matrices = val_matrix_to_adjacency_matrices(results["val_matrix"], results["p_matrix"], alpha=0.05)
    
    return adj_matrices


def val_matrix_to_adjacency_matrices(val_matrix, p_matrix, alpha=0.05):
    """
    Convert PCMCI val_matrix to adjacency_matrices format.
    
    Parameters:
    val_matrix (np.array): The val_matrix from PCMCI results
    p_matrix (np.array): The p_matrix from PCMCI results
    alpha (float): Significance level for p-values (default: 0.05)
    
    Returns:
    list: A list of adjacency matrices (same as the VAR-LiNGAM format)
    """
    
    # Get the dimensions
    n_vars, _, n_lags = val_matrix.shape
    
    # Initialize the list of adjacency matrices
    adjacency_matrices = []
    
    # For each lag (including contemporaneous effects)
    for lag in range(n_lags):
        # Create a mask for significant relationships
        significant_mask = (p_matrix[:,:,lag] < alpha)
        
        # Get the values for this lag and apply the mask
        adj_matrix = np.where(significant_mask, val_matrix[:,:,lag], 0)# Get the values for this lag and apply the mask
        
        # Append to the list
        adjacency_matrices.append(adj_matrix)
    
    return adjacency_matrices


def run_varlingam_bootstrap(data, lags=5, n_sampling=10, variance_threshold=0.8, occurrence_threshold=0.5):
    """
    Run VAR-LiNGAM bootstrap on the given data.
    
    Args:
    data (pd.DataFrame): Time series data.
    lags (int): Number of lags to include. Default is 5.
    n_sampling (int): Number of bootstrap samples. Default is 10.
    variance_threshold (float): Threshold for variance to consider an edge unstable. Default is 0.8.
    occurrence_threshold (float): Minimum fraction of occurrences for an edge to be considered. Default is 0.5.
    
    Returns:
    list: List of filtered adjacency matrices for each lag.
    """
    model = lingam.VARLiNGAM(lags=lags, prune=True)
    bootstrap_result = model.bootstrap(data, n_sampling=n_sampling)
    
    filtered_matrices = variance_filtered_adjacency_matrix(
        bootstrap_result.adjacency_matrices_,
        variance_threshold=variance_threshold,
        occurrence_threshold=occurrence_threshold
    )
    
    return filtered_matrices

def variance_filtered_adjacency_matrix(adjacency_matrices_list, variance_threshold=0.8, occurrence_threshold=0.5):
    """
    Calculate an adjacency matrix with high-variance edges set to zero.
    
    Args:
    adjacency_matrices_list (list): List of adjacency matrices from bootstrap samples.
    variance_threshold (float): Threshold for variance to consider an edge unstable.
    occurrence_threshold (float): Minimum fraction of occurrences for an edge to be considered.
    
    Returns:
    list: Filtered adjacency matrices for each lag.
    """
    adjacency_matrices = np.array(adjacency_matrices_list)
    
    edge_medians = np.median(adjacency_matrices, axis=0)
    edge_variances = np.var(adjacency_matrices, axis=0)
    edge_occurrences = np.mean(np.abs(adjacency_matrices) > 0.05, axis=0)
    
    filtered_adjacency = edge_medians.copy()
    unstable_mask = (edge_variances > variance_threshold) | (edge_occurrences < occurrence_threshold)
    filtered_adjacency[unstable_mask] = 0
    
    return reshape_adjacency_matrix(filtered_adjacency)


def reshape_adjacency_matrix(avg_adjacency_matrices):
    """
    Reshape the flattened adjacency matrix into separate matrices for each lag.
    
    Args:
    avg_adjacency_matrices (np.ndarray): Flattened adjacency matrix.
    
    Returns:
    list: List of adjacency matrices, one for each lag.
    """
    rows, cols = avg_adjacency_matrices.shape
    num_vars = rows
    num_matrices = cols // num_vars

    reshaped_matrices = []
    for i in range(num_matrices):
        start_col = i * num_vars
        end_col = (i + 1) * num_vars
        lag_matrix = avg_adjacency_matrices[:, start_col:end_col]
        reshaped_matrices.append(lag_matrix)
    
    return reshaped_matrices

def grid_search_bootstrap_varlingam(data, true_matrices, param_grid):
    """
    Perform grid search for VAR-LiNGAM bootstrap method.
    
    Args:
    data (pd.DataFrame): Time series data.
    true_matrices (list): List of true adjacency matrices.
    param_grid (dict): Dictionary of parameters to search over.
    
    Returns:
    dict: Results of the grid search.
    """
    
    best_score = float('inf')
    best_params = {}
    best_matrices = None
    
    for lags in param_grid.get('lags', [5]):
        for n_sampling in param_grid.get('n_sampling', [10]):
            for variance_threshold in param_grid.get('variance_threshold', [0.8]):
                for occurrence_threshold in param_grid.get('occurrence_threshold', [0.5]):
                    bootstrap_matrices = run_varlingam_bootstrap(
                        data, lags, n_sampling, variance_threshold, occurrence_threshold
                    )
                    score = evaluate_causal_matrices(true_matrices, bootstrap_matrices)['f1']
                    
                    if score < best_score:
                        best_score = score
                        best_params = {
                            'lags': lags,
                            'n_sampling': n_sampling,
                            'variance_threshold': variance_threshold,
                            'occurrence_threshold': occurrence_threshold
                        }
                        best_matrices = bootstrap_matrices
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'best_matrices': best_matrices
    }

def run_tcdf(data, cuda=False, nrepochs=1000, kernel_size=4, levels=1, 
             loginterval=500, learningrate=0.01, optimizername='Adam',
             seed=1111, dilation_c=4, significance=0.8):
    """Run TCDF analysis on numpy array data"""
    
    # Create temporary CSV with numerical column names
    df = pd.DataFrame(data, columns=[f'x{i}' for i in range(data.shape[1])])
    temp_filename = 'temp_tcdf.csv'
    df.to_csv(temp_filename, index=False)
    
    # Run TCDF analysis
    allcauses = {}
    alldelays = {}
    
    for i in range(data.shape[1]):
        col_name = f'x{i}'
        causes, causeswithdelay, _, _ = TCDF.findcauses(
            col_name, cuda=cuda, epochs=nrepochs, kernel_size=kernel_size, 
            layers=levels, log_interval=loginterval, lr=learningrate,
            optimizername=optimizername, seed=seed, dilation_c=dilation_c,
            significance=significance, file=temp_filename
        )
        allcauses[i] = causes
        alldelays.update(causeswithdelay)
    
    # Clean up
    os.remove(temp_filename)
    
    # Convert to adjacency matrices
    adj_matrices = tcdf_to_matrices(alldelays, data.shape[1])
    
    return adj_matrices

def tcdf_to_matrices(alldelays, n_variables):
    """
    Convert TCDF delays output to adjacency matrices
    
    Parameters:
    -----------
    alldelays : dict
        Dictionary where keys are (target, cause) tuples and values are delays
    n_variables : int
        Number of variables in the dataset
        
    Returns:
    --------
    list
        List of adjacency matrices, where index represents the time lag
    """
    max_delay = max(alldelays.values()) if alldelays else 0
    adj_matrices = [np.zeros((n_variables, n_variables)) for _ in range(max_delay + 1)]
    
    for (target, cause), delay in alldelays.items():
        adj_matrices[delay][target][cause] = 1
            
    return adj_matrices

def run_rcd_varlingam(data, n_splits=5, consistency_threshold=0.4, 
                     variability_threshold=0.4, adjustment_weight=0, **varlingam_params):
    """
    Run RCD-VARLiNGAM using the generic RCD framework
    """
    
    # Set default VARLiNGAM parameters if not provided
    params = {'lags': 3}
    params.update(varlingam_params)
    
    return run_rcd(data, run_varlingam, n_splits=n_splits,
                  consistency_threshold=consistency_threshold,
                  variability_threshold=variability_threshold,
                  adjustment_weight=adjustment_weight,
                  **params)

def grid_search_rcd_varlingam(data, true_matrices, param_grid=None, varlingam_params=None):
    """
    Perform grid search for RCD-VARLiNGAM parameters.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input time series data
    true_matrices : list
        True adjacency matrices for evaluation
    param_grid : dict, optional
        Grid of RCD parameters to search
    varlingam_params : dict, optional
        Additional parameters for VARLiNGAM
        
    Returns:
    --------
    dict
        Best parameters, score, and matrices
    """
    
    # Set default VARLiNGAM parameters if not provided
    base_params = {'lags': 3}
    if varlingam_params:
        base_params.update(varlingam_params)
    
    return grid_search_rcd(
        data=data,
        true_matrices=true_matrices,
        base_method=run_varlingam,
        param_grid=param_grid,
        method_params=base_params
    )

def run_rcd_pcmci(data, n_splits=7, consistency_threshold=0.7,
                  variability_threshold=0.4, adjustment_weight=0, **pcmci_params):
    """
    Run RCD-PCMCI using the generic RCD framework
    """
    
    # Set default PCMCI parameters if not provided
    params = {'alpha': 0.05, 'tau_max': 3}
    params.update(pcmci_params)
    
    return run_rcd(data, run_pcmci, n_splits=n_splits,
                  consistency_threshold=consistency_threshold,
                  variability_threshold=variability_threshold,
                  adjustment_weight=adjustment_weight,
                  **params)

def grid_search_rcd_pcmci(data, true_matrices, param_grid=None, pcmci_params=None):
    """
    Perform grid search for RCD-PCMCI parameters.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input time series data
    true_matrices : list
        True adjacency matrices for evaluation
    param_grid : dict, optional
        Grid of RCD parameters to search
    pcmci_params : dict, optional
        Additional parameters for PCMCI
        
    Returns:
    --------
    dict
        Best parameters, score, and matrices
    """
    
    # Set default PCMCI parameters if not provided
    base_params = {'alpha': 0.05, 'tau_max': 3}
    if pcmci_params:
        base_params.update(pcmci_params)
    
    return grid_search_rcd(
        data=data,
        true_matrices=true_matrices,
        base_method=run_pcmci,
        param_grid=param_grid,
        method_params=base_params
    )