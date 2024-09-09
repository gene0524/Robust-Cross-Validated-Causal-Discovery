import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm


def test_statistical_equilibrium(AB_data):
    """
    Test for statistical equilibrium across different time periods in the ABM data.
    
    Args:
        AB_data (np.array): ABM data with shape (M, T, K) where M is the number of simulations,
                            T is the number of time steps, and K is the number of variables.
    
    Returns:
        np.array: Proportion of p-values > 0.05 for each variable.
    """
    M, T, K = AB_data.shape
    p_values = np.zeros((K, T*(T-1)//2))
    
    for k in range(K):
        idx = 0
        for i in range(T):
            for j in range(i+1, T):
                _, p = stats.ks_2samp(AB_data[:, i, k], AB_data[:, j, k])
                p_values[k, idx] = p
                idx += 1
    
    return np.mean(p_values > 0.05, axis=1)

def test_ergodicity(AB_data):
    """
    Test for ergodicity in the ABM data.
    
    Args:
        AB_data (np.array): ABM data with shape (M, T, K) where M is the number of simulations,
                            T is the number of time steps, and K is the number of variables.
    
    Returns:
        np.array: Proportion of p-values > 0.05 for each variable.
    """
    M, T, K = AB_data.shape
    p_values = np.zeros((K, M*T))
    
    for k in range(K):
        idx = 0
        for i in range(M):
            for j in range(T):
                _, p = stats.ks_2samp(AB_data[i, :, k], AB_data[:, j, k])
                p_values[k, idx] = p
                idx += 1

    return np.mean(p_values > 0.05, axis=1)

def test_linearity(data):
    """
    Test for linearity in the data using Ramsey's RESET test.
    
    Args:
        data (np.array): Data with shape (T, K) where T is the number of time steps
                         and K is the number of variables.
    
    Returns:
        np.array: Array of linearity scores (1 - p_value) for each variable.
    """
    T, K = data.shape
    linearity_scores = np.zeros(K)
    
    for k in range(K):
        y = data[:, k]
        X = np.column_stack((np.ones(T), data[:, [i for i in range(K) if i != k]]))
        
        model = sm.OLS(y, X).fit()
        y_hat = model.predict(X)
        X_aug = np.column_stack((X, y_hat**2, y_hat**3))
        
        model_aug = sm.OLS(y, X_aug).fit()
        f_statistic = ((model.ssr - model_aug.ssr) / 2) / (model_aug.ssr / (T - K - 3))
        p_value = 1 - stats.f.cdf(f_statistic, 2, T - K - 3)
        linearity_scores[k] = 1 - p_value  # 將p值轉換為線性性得分
    
    return linearity_scores

def test_stationarity(data):
    """
    Test for stationarity using the Augmented Dickey-Fuller test.
    
    Args:
        data (np.array): Data with shape (T, K) where T is the number of time steps
                         and K is the number of variables.
    
    Returns:
        np.array: Array of stationarity scores (1 - p_value) for each variable.
    """
    _, K = data.shape
    stationarity_scores = np.zeros(K)
    
    for k in range(K):
        result = adfuller(data[:, k])
        stationarity_scores[k] = 1 - result[1]  # 將p值轉換為穩定性得分
    
    return stationarity_scores

def analyze_rw_properties(data):
    """
    Analyze properties of real-world data.
    
    Args:
        data (np.array): Data with shape (T, K) where T is the number of time steps
                         and K is the number of variables.
    
    Returns:
        dict: Dictionary containing analysis results for various properties.
    """
    properties = {}
    
    properties['linearity'] = test_linearity(data)
    properties['stationarity'] = test_stationarity(data)
    
    return properties

def analyze_ab_properties(AB_data):
    """
    Analyze properties specific to ABM data.
    
    Args:
        AB_data (np.array): ABM data with shape (M, T, K) where M is the number of simulations,
                            T is the number of time steps, and K is the number of variables.
    
    Returns:
        dict: Dictionary containing analysis results for ABM-specific properties.
    """
    M, T, K = AB_data.shape
    properties = {}
    
    # Statistical equilibrium test
    properties['statistical_equilibrium'] = test_statistical_equilibrium(AB_data)
    
    # Ergodicity test
    properties['ergodicity'] = test_ergodicity(AB_data)
    
    # Linearity and stationarity tests for each simulation
    linearity_scores = np.zeros((M, K))
    stationarity_scores = np.zeros((M, K))
    
    for m in range(M):
        linearity_scores[m] = test_linearity(AB_data[m])
        stationarity_scores[m] = test_stationarity(AB_data[m])
    
    # Compute mean of scores across simulations
    properties['linearity'] = np.mean(linearity_scores, axis=0)
    properties['stationarity'] = np.mean(stationarity_scores, axis=0)
    
    return properties