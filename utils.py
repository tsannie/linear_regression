import numpy as np

def r_squared(target: np.ndarray, model: np.ndarray) -> float:
    """Compute the R-squared value of the model"""
    u = ((target - model) ** 2 ).sum()
    v = ((target - target.mean()) ** 2).sum()
    return 1 - u / v

def cost_function(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    """Compute the cost function"""
    m = y.size
    F = X.dot(theta)
    return (1 / (2 * m)) * np.sum((F - y) ** 2)

def normalize_data(X: np.ndarray) -> np.ndarray:
    """Normalize the data"""
    return (X - np.min(X)) / (np.max(X) - np.min(X))

def denormalize_theta(theta: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Denormalize the theta"""
    gap_y = np.max(y) - np.min(y)
    gap_x = np.max(x) - np.min(x)
    a = theta[0] * gap_y / gap_x
    b = theta[1] * gap_y + np.min(y) - a * np.min(x)
    return np.array([a, b]).reshape(-1, 1)

def gradient(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Compute the gradient"""
    m = y.size
    F = X.dot(theta)
    return (1 / m) * X.T.dot(F - y)
