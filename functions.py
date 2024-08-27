import numpy as np
from sklearn.preprocessing import StandardScaler

def normalization(X: np.ndarray):
  """
    Rescale the input array X using StandardScaler module of Scikitlearn library
    Args:
        X (numpy.ndarray): The input array to be rescaled.
    Returns:
        numpy.ndarray: The rescaled version of the input array X.
    """
  scaler = StandardScaler()
  scaler.fit(X)
  return scaler.transform(X)

    
def generate_randomly_missing(X , missing_rate):
    """
    Creates a randomly missing mask for the input data.

    Args:
        data (np.ndarray): The input data.
        missing_rate (float): The ratio of missing values to create.

    Returns:
        np.ndarray: An array with the same shape as `data` where missing values are marked as NaN.
    """

    X_copy=np.copy(X)
    XmShape = X_copy.shape
    na_id = np.random.randint(0, X_copy.size, round(missing_rate * X_copy.size))
    X_nan = X_copy.flatten()
    X_nan[na_id] = np.nan
    X_nan = X_nan.reshape(XmShape)
    return X_nan


def solving(a,b,c,d):
    roots = np.roots([a, b, c, d])
    real_roots = roots[np.isreal(roots)]
    if len(real_roots) == 1:
        return real_roots[0].real