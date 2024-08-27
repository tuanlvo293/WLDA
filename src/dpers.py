import numpy as np
from functions import solving

def sig_estimate(f1,f2):
    """
    Estimates the variances and covariance for the given dataset.

    Parameters:
    X (np.ndarray): A 2D numpy array where each column is a data sample [x0, x1].
    mus0 (float): The mean of the first variable (x0).
    mus1 (float): The mean of the second variable (x1).

    Returns:
    tuple: A tuple containing:
        - sig11 (float): The variance of x0.
        - sig22 (float): The variance of x1.
        - sig12 (float): The covariance between x0 and x1.
    """
    is_finite_1 = np.isfinite(f1)
    is_finite_2 = np.isfinite(f2)
    valid_indices = is_finite_1 & is_finite_2

    m = np.sum(valid_indices)
    n = np.sum(is_finite_1 & ~is_finite_2)
    l = np.sum(~is_finite_1 & is_finite_2)

    diff1 = f1 - np.nanmean(f1)
    diff2 = f2 - np.nanmean(f2)

    s11 = np.sum(diff1[valid_indices] ** 2)
    s22 = np.sum(diff2[valid_indices] ** 2)
    s12 = np.sum(diff1[valid_indices] * diff2[valid_indices])

    sig11 = s11 + np.sum(diff1[is_finite_1 & ~is_finite_2] ** 2)
    sig22 = s22 + np.sum(diff2[~is_finite_1 & is_finite_2] ** 2)

    sig11 /= (m + n)
    sig22 /= (m + l)

    sig12 = solving(-m, s12, (m * sig11 * sig22 - s22 * sig11 - s11 * sig22), s12 * sig11 * sig22)

    return sig11,sig22,sig12

def DPER(X):
    """
    Estimates the covariance matrix for the dataset X, accounting for missing data.

    Parameters:
    X (np.ndarray): A 2D numpy array where rows are observations and columns are variables.

    Returns:
    np.ndarray: The estimated covariance matrix.
    """
    X = np.array(X)
    num_features = X.shape[1]
    sig = np.zeros((num_features, num_features))
    for i in range(num_features):
        for j in range(i):
            temp = sig_estimate(X[:, j], X[:, i])
            sig[j, j] = temp[0]
            sig[i, i] = temp[1]
            sig[j, i] = sig[i, j] = temp[2]
    return sig

def sigma_m(X, y):
    G = len(np.unique(y))
    mus = [np.nanmean(X[y == g], axis=0) for g in range(G)]
    res = np.zeros(8)  # [m, n, l, s11, s12, s22, sig11, sig22]

    for g in range(G):
        Xg = X[y == g]
        valid_1 = np.isfinite(Xg[:, 0])
        valid_2 = np.isfinite(Xg[:, 1])

        m = np.sum(valid_1 & valid_2)
        n = np.sum(valid_1 & ~valid_2)
        l = np.sum(~valid_1 & valid_2)

        diff_1 = Xg[:, 0] - mus[g][0]
        diff_2 = Xg[:, 1] - mus[g][1]

        s11 = np.sum(diff_1[valid_1 & valid_2] ** 2)
        s22 = np.sum(diff_2[valid_1 & valid_2] ** 2)
        s12 = np.sum(diff_1[valid_1 & valid_2] * diff_2[valid_1 & valid_2])

        sig11 = s11 + np.sum(diff_1[valid_1 & ~valid_2] ** 2)
        sig22 = s22 + np.sum(diff_2[~valid_1 & valid_2] ** 2)

        res += np.array([m, n, l, s11, s12, s22, sig11, sig22])

    m, n, l, s11, s12, s22, sig11, sig22 = res
    sig11 /= (m + n)
    sig22 /= (m + l)

    sig12 = solving(-m, s12, (m * sig11 * sig22 - s22 * sig11 - s11 * sig22), s12 * sig11 * sig22)

    return sig11, sig22, sig12

def DPERm(X, y):
    num_features = X.shape[1]
    sig = np.zeros((num_features, num_features))

    for i in range(num_features):
        for j in range(i):
            temp = sigma_m(X[:, [j, i]], y)
            sig[j, j] = temp[0]
            sig[i, i] = temp[1]
            sig[j, i] = sig[i, j] = temp[2]

    return sig
