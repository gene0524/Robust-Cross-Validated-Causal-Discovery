import numpy as np

def preprocess_data(data, columns, log_vars=None, percent_vars=None):

    if log_vars is None:
        log_vars = []
    if percent_vars is None:
        percent_vars = []

    # Convert column names to indices
    log_indices = [columns.index(var) for var in log_vars if var in columns]
    percent_indices = [columns.index(var) for var in percent_vars if var in columns]

    processed_data = data.copy()

    # Apply log transformation
    for idx in log_indices:
        processed_data[:, idx] = np.log(processed_data[:, idx])
    
    # Convert to percentages
    for idx in percent_indices:
        if abs(processed_data[:, idx].mean()) < 1:
            processed_data[:, idx] *= 100

    return np.round(processed_data, 3)