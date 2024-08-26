import numpy as np
from src.models.lingam_master import lingam
from src.models.tigramite_master.tigramite.pcmci import PCMCI
from src.models.tigramite_master.tigramite.independence_tests.parcorr import ParCorr
from src.models.tigramite_master.tigramite import data_processing as pp
from src.causal_matrix_evaluation import evaluate_causal_matrices
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
    return results


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
    from src.causal_matrix_evaluation import evaluate_causal_matrices
    
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
                    score = evaluate_causal_matrices(true_matrices, bootstrap_matrices)['fro']
                    
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


def grid_search_varlingam_bootstrap(data, true_matrices, param_grid=None):
    """
    Perform grid search to find the best parameters for VARLiNGAM bootstrap().

    Args:
    data (np.array): The input time series data.
    true_matrices (list): List of true causal matrices.
    param_grid (dict, optional): Dictionary with parameters names (str) as keys and lists of parameter settings to try as values.

    Returns:
    dict: Best parameters, best score, and best matrices.
    """
    if param_grid is None:
        param_grid = {
            'lags': range(1, 6),
            'n_sampling': range(10, 101, 20),
            'variance_threshold': np.arange(0.1, 1.1, 0.2),
            'occurrence_threshold': np.arange(0.1, 1.0, 0.2)
        }

    param_combinations = list(product(*param_grid.values()))
    best_score = float('inf')
    best_params = None
    best_matrices = None

    for params in param_combinations:
        current_params = dict(zip(param_grid.keys(), params))
        
        bootstrap_matrices = run_varlingam_bootstrap(data, **current_params)
        
        evaluation_results = evaluate_causal_matrices(true_matrices, bootstrap_matrices)
        current_score = evaluation_results['fro']  # Using Frobenius norm as the score

        if current_score < best_score:
            best_score = current_score
            best_params = current_params
            best_matrices = bootstrap_matrices

    return {
        'best_params': best_params,
        'best_score': best_score,
        'best_matrices': best_matrices
    }
