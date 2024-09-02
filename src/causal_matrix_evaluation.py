import numpy as np
from sklearn.metrics import f1_score

def frobenius_norm(A, B):
    return np.linalg.norm(A - B, 'fro')

def structural_hamming_distance(A, B):
    return np.sum(np.abs(np.sign(A) - np.sign(B)))

def calculate_f1(A, B):
    true = (np.abs(A) > 1e-6).flatten()
    pred = (np.abs(B) > 1e-6).flatten()
    return f1_score(true, pred)

def calculate_f1_directed(A, B):
    true = np.sign(A).flatten()
    pred = np.sign(B).flatten()
    return f1_score(true, pred, labels=[-1, 1], average='micro')

def evaluate_causal_matrices(true_matrices, est_matrices):
    results = {}
    
    # Record the number of lags
    results['true_lags'] = len(true_matrices)
    results['est_lags'] = len(est_matrices)
    results['extra_lags'] = max(0, len(est_matrices) - len(true_matrices))
    results['missing_lags'] = max(0, len(true_matrices) - len(est_matrices))
    
    # Combine true matrices
    true_combined = np.hstack(true_matrices)
    
    # Combine estimated matrices, padding with zeros if necessary
    if len(est_matrices) < len(true_matrices):
        # Pad with zero matrices if estimated lags are fewer than true lags
        num_vars = est_matrices[0].shape[0]
        zero_matrices = [np.zeros((num_vars, num_vars)) for _ in range(len(true_matrices) - len(est_matrices))]
        est_matrices = np.vstack((est_matrices, zero_matrices))
        est_combined = np.hstack(est_matrices + zero_matrices)
    else:
        # Use only up to the number of true lags
        est_combined = np.hstack(est_matrices[:len(true_matrices)])
    
    # print("True combined: \n", true_combined)
    # print("Est combined: \n", est_combined)
    
    # Check if true and estimated matrices have the same shape
    if len(true_combined) !=  len(est_combined):
        print("Numbers of Variable Mismatch. True and estimated matrices have different shapes")
        return None
    
    # Calculate metrics for the combined matrices
    results['fro'] = round(frobenius_norm(true_combined, est_combined), 3)
    results['shd'] = int(structural_hamming_distance(true_combined, est_combined))
    results['f1'] = round(calculate_f1(true_combined, est_combined), 3)
    results['f1_directed'] = round(calculate_f1_directed(true_combined, est_combined), 3)

    # Record the number of true edges
    results['num_true_edges'] = np.sum(np.abs(np.sign(true_combined)))
    
    return results

def interpret_evaluation_metrics(results):
    interpretations = {}
    
    # Interpret Frobenius norm
    fro = results['fro']
    if fro < 0.3:
        interpretations['Frobenius Norm'] = f"Excellent: {fro:.3f}. Very close match between true and estimated matrices."
    elif fro < 0.7:
        interpretations['Frobenius Norm'] = f"Good: {fro:.3f}. Reasonable match between true and estimated matrices."
    elif fro < 1.0:
        interpretations['Frobenius Norm'] = f"Fair: {fro:.3f}. Some discrepancies between true and estimated matrices."
    else:
        interpretations['Frobenius Norm'] = f"Poor: {fro:.3f}. Large discrepancies between true and estimated matrices."
    
    # Interpret Structural Hamming Distance 
    shd = results['shd']
    true_edges = np.prod(results['num_true_edges'])
    normalized_shd = shd / true_edges if true_edges > 0 else 0
    # print("SHD: ", shd)
    # print("True Edges: ", true_edges)
    # print("Normalized SHD: ", normalized_shd)
    if normalized_shd < 0.3:
        interpretations['Structural Hamming Distance'] = f"Excellent: {shd} ({normalized_shd:.2%}). Very few structural differences."
    elif normalized_shd < 0.7:
        interpretations['Structural Hamming Distance'] = f"Good: {shd} ({normalized_shd:.2%}). Some structural differences."
    elif normalized_shd < 1.0:
        interpretations['Structural Hamming Distance'] = f"Fair: {shd} ({normalized_shd:.2%}). Significant structural differences."
    else:
        interpretations['Structural Hamming Distance'] = f"Poor: {shd} ({normalized_shd:.2%}). Many structural differences."
    
    # Interpret F1 Score
    f1 = results['f1']
    if f1 > 0.9:
        interpretations['f1'] = f"Excellent: {f1:.3f}. Very high accuracy in identifying causal relationships."
    elif f1 > 0.7:
        interpretations['f1'] = f"Good: {f1:.3f}. Good accuracy in identifying causal relationships."
    elif f1 > 0.5:
        interpretations['f1'] = f"Fair: {f1:.3f}. Moderate accuracy in identifying causal relationships."
    else:
        interpretations['f1'] = f"Poor: {f1:.3f}. Low accuracy in identifying causal relationships."
    
    # Interpret Directed F1 Score
    f1_directed = results['f1_directed']
    if f1_directed > 0.9:
        interpretations['F1→'] = f"Excellent: {f1_directed:.3f}. Very high accuracy in identifying directed causal relationships."
    elif f1_directed > 0.7:
        interpretations['F1→'] = f"Good: {f1_directed:.3f}. Good accuracy in identifying directed causal relationships."
    elif f1_directed > 0.5:
        interpretations['F1→'] = f"Fair: {f1_directed:.3f}. Moderate accuracy in identifying directed causal relationships."
    else:
        interpretations['F1→'] = f"Poor: {f1_directed:.3f}. Low accuracy in identifying directed causal relationships."
    
    # Interpret lag estimation
    if results['extra_lags'] == 0 and results['missing_lags'] == 0:
        interpretations['lags'] = f"Perfect: Correctly estimated {results['true_lags']} lags."
    elif results['extra_lags'] > 0:
        interpretations['lags'] = f"Overestimation: Estimated {results['est_lags']} lags, with {results['extra_lags']} extra lags."
    else:
        interpretations['lags'] = f"Underestimation: Estimated {results['est_lags']} lags, lack {results['missing_lags']} lags."
    
    return interpretations