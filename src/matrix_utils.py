import numpy as np
import os

def read_matrices_from_csv(filepath):
    """
    Read multiple matrices from a CSV file where matrices are separated by double newlines.
    
    Args:
        filepath: Path to the CSV file
    Returns:
        List of numpy arrays representing the matrices
    """
    try:
        with open(filepath, 'r') as f:
            content = f.read().strip()
            
        if not content:
            print(f"Warning: Empty file: {filepath}")
            return None
            
        # Split content into matrix strings
        matrix_strings = content.split('\n\n')
        
        # Convert each matrix string to numpy array
        matrices = []
        for matrix_string in matrix_strings:
            matrix = np.array([
                list(map(float, row.split(',')))
                for row in matrix_string.strip().split('\n')
            ])
            matrices.append(matrix)
                
        return matrices
        
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return None

def save_matrices(matrices, filepath):
    """
    Save multiple matrices to a file.
    
    Args:
        matrices: List of matrices to save
        filepath: Output file path
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            for i, matrix in enumerate(matrices):
                np.savetxt(f, matrix, delimiter=',', fmt='%.3f')
                if i < len(matrices) - 1:
                    f.write('\n')
        return True
        
    except Exception as e:
        print(f"Error saving matrices: {str(e)}")
        return False

def get_summary_matrix(matrices):
    """
    Convert time-lagged adjacency matrices to a summary causal graph.
    
    Args:
        matrices: List of numpy arrays representing adjacency matrices
                First matrix is contemporaneous effects,
                Following matrices are lagged effects
    Returns:
        A single matrix representing the summary causal graph
    """
    if len(matrices) == 0:
        raise ValueError("Input must be a non-empty list of adjacency matrices")
    
    n_vars = matrices[0].shape[0]
    summary = np.zeros((n_vars, n_vars))
    
    # Process all matrices (contemporaneous and lagged)
    for matrix in matrices:
        for i in range(n_vars):
            for j in range(n_vars):
                if matrix[i, j] != 0:
                    # Update if no relationship yet or if current is stronger
                    if summary[i, j] == 0 or abs(matrix[i, j]) > abs(summary[i, j]):
                        summary[i, j] = 1 if matrix[i, j] > 0 else -1
    
    return summary