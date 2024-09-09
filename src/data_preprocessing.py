import numpy as np
import pandas as pd
from scipy import interpolate

def preprocess_data(data, columns=None, log_vars=None, percent_vars=None, inf_nan_method='interpolate'):
    """
    Preprocess the input data by handling inf/nan values, applying log transformation,
    and converting to percentages where specified.

    Args:
        data (np.array): Input data to be processed
        columns (list): List of column names
        log_vars (list): Variables to be log-transformed
        percent_vars (list): Variables to be converted to percentages
        inf_nan_method (str): Method to handle inf/nan values ('interpolate', 'ffill', 'bfill', or 'mean')

    Returns:
        np.array: Processed data
    """
    if log_vars is None:
        log_vars = []
    if percent_vars is None:
        percent_vars = []

    # Convert column names to indices
    log_indices = [columns.index(var) for var in log_vars if var in columns]
    percent_indices = [columns.index(var) for var in percent_vars if var in columns]

    processed_data = data.copy()
    
    # Handle inf and nan values
    processed_data = handle_inf_nan(processed_data, method=inf_nan_method)

    # Apply log transformation
    for idx in log_indices:
        processed_data[:, idx] = np.log(processed_data[:, idx])
    
    # Convert to percentages
    for idx in percent_indices:
        if abs(processed_data[:, idx].mean()) < 1:
            processed_data[:, idx] *= 100

    return np.round(processed_data, 3)

def handle_inf_nan(data, method='interpolate'):
    problem_mask = np.isnan(data) | np.isinf(data)
    
    if problem_mask.any():
        problem_rows = np.where(problem_mask.any(axis=1))[0]
        print(f"Warning: Data contains inf or nan values at rows: {problem_rows}")
        
        if method == 'interpolate':
            for col in range(data.shape[1]):
                mask = problem_mask[:, col]
                if mask.any():
                    good = ~mask
                    f = interpolate.interp1d(np.flatnonzero(good), data[good, col], 
                                             bounds_error=False, fill_value='extrapolate')
                    data[mask, col] = f(np.flatnonzero(mask))
        elif method in ['ffill', 'bfill']:
            data = pd.DataFrame(data).fillna(method=method).values
        elif method == 'mean':
            col_means = np.nanmean(data, axis=0)
            data[problem_mask] = np.take(col_means, np.where(problem_mask)[1])
        else:
            raise ValueError("Invalid method. Choose 'interpolate', 'ffill', 'bfill', or 'mean'.")
        
        print(f"Handled {np.sum(problem_mask)} inf/nan values using {method} method.")
    
    return data