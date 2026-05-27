import numpy as np


def compute_rmse(actual, predicted):
    """Root Mean Square Error (paper Equation 26)."""
    return np.sqrt(np.mean((predicted - actual) ** 2))


def compute_mae(actual, predicted):
    """Mean Absolute Error (paper Equation 27)."""
    return np.mean(np.abs(predicted - actual))


def compute_mape(actual, predicted):
    """Mean Absolute Percentage Error (paper Equation 28).
    Guard against zero division with 1e-8.
    """
    return 100.0 * np.mean(np.abs((predicted - actual) / (np.abs(actual) + 1e-8)))


def compute_all_metrics(actual, predicted):
    """Compute RMSE, MAE, MAPE and return as dict."""
    return {
        'rmse': compute_rmse(actual, predicted),
        'mae': compute_mae(actual, predicted),
        'mape': compute_mape(actual, predicted),
    }
