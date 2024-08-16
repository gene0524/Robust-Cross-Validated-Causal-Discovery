import numpy as np
from src.models.lingam_master import lingam
from src.models.tigramite_master.tigramite.pcmci import PCMCI
from src.models.tigramite_master.tigramite.independence_tests.parcorr import ParCorr
from src.models.tigramite_master.tigramite import data_processing as pp

def run_varlingam(data, lags=5):
    """
    Run VAR-LiNGAM on the given data.
    
    :param data: pandas DataFrame
    :param lags: number of lags to include (default: 5)
    :return: VARLiNGAM results
    """
    model = lingam.VARLiNGAM(lags=lags, prune=True)
    results = model.fit(data)
    return results


def run_varlingam_bootstrap(data, lags=5):
    """
    Run VAR-LiNGAM on the given data.
    
    :param data: pandas DataFrame
    :param lags: number of lags to include (default: 5)
    :return: VARLiNGAM results
    """
    model = lingam.VARLiNGAM(lags=lags, prune=True)
    bootstrap_result = model.bootstrap(data, n_sampling=10)
    
    # Calculate the average adjacency matrices
    filtered_avg_adjacency_matrices = variance_filtered_adjacency_matrix(bootstrap_result.adjacency_matrices_)

    return filtered_avg_adjacency_matrices


def variance_filtered_adjacency_matrix(adjacency_matrices_list, variance_threshold=0.8, occurrence_threshold=0.5):
    """
    Calculate an adjacency matrix with high-variance edges set to zero.
    
    :param adjacency_matrices_list: List of adjacency matrices from bootstrap samples
    :param variance_threshold: Threshold for variance to consider an edge unstable
    :param occurrence_threshold: Minimum fraction of occurrences for an edge to be considered
    :return: Filtered adjacency matrix
    """

    # Convert to numpy array if it's not already
    adjacency_matrices = np.array(adjacency_matrices_list)

    # Calculate median and variance for each edge
    edge_medians = np.median(adjacency_matrices, axis=0)
    edge_variances = np.var(adjacency_matrices, axis=0)

    # Calculate the occurrence frequency of each edge
    edge_occurrences = np.mean(abs(adjacency_matrices) > 0.05, axis=0)

    # Initialize the filtered adjacency matrix with the median values
    filtered_adjacency = edge_medians.copy()

    # Set edges to zero if they have high variance or low occurrence
    unstable_mask = (edge_variances > variance_threshold) | (edge_occurrences < occurrence_threshold)
    filtered_adjacency[unstable_mask] = 0

    # Reshape the filtered adjacency matrix
    return reshape_adjacency_matrix(filtered_adjacency)


def reshape_adjacency_matrix(avg_adjacency_matrices):
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


def run_pcmci(data, columns=None, alpha=0.05, tau_max=1):
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