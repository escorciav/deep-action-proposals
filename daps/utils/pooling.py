import numpy as np


def pyramid1d(x, levels=0, pool_type='mean', norm=True, unit=False):
    """Compute a 1d pyramid representation of a feature vector

    Parameters
    ----------
    x : ndarray
        [m x d] array of features. m is the number of features and d is the
        dimensionality of the feature space.
    levels : int
        Number of levels of the pyramid representation.
    pool_type : str
        Pooling strategy over a bunch of features.
    norm : bool
        Normalize each region before concatenate them.
    unit : bool
        Normalize the final input vector.

    Outputs
    -------
    [d * (2**(levels + 1) - 1)] ndarray with pyramid represetantion of x.

    """
    m, d = x.shape
    arr = [np.empty(d) for i in range((2**(levels + 1) - 1))]
    pool_type = pool_type.lower()

    idx = 0
    for i in range(levels + 1):
        n = 2 ** i
        edges = np.ones(n + 1, dtype=int) * 1.0 / n
        edges[0] = 0
        edges = np.round(np.cumsum(edges) * m).astype(int)
        for j in range(n):
            if pool_type == 'mean':
                arr[idx][...] = x[edges[j]:edges[j + 1], :].mean(axis=0)
            elif pool_type == 'max':
                arr[idx][...] = x[edges[j]:edges[j + 1], :].max(axis=0)
            else:
                raise ValueError('Unknown pooling type {}'.format(pool_type))

            if norm:
                feat_norm = np.sqrt((arr[idx] ** 2).sum())
                if feat_norm == 0:
                    feat_norm = 1.0
                arr[idx] /= feat_norm

            idx += 1
    pyr_feat = np.hstack(arr)
    if unit:
        return pyr_feat / (2**(levels + 1) - 1)
    return pyr_feat


def concat1d(x, n=8, pool_type='mean', norm=True, unit=False):
    """Compute a 1d pyramid representation of a feature vector
    Parameters
    ----------
    x : ndarray.
        [m x d] array of features. m is the number of features and d is the
        dimensionality of the feature space.
    n : int
        Number of chunks.
    pool_type : str.
        Pooling strategy over a bunch of features.
    norm : bool.
        Normalize each region before concatenate them.
    unit : bool.
        Normalize the final input vector.
    Outputs
    -------
    [d * n] ndarray with concat feature of x.
    """
    m, d = x.shape
    arr = [np.empty(d) for i in range(n)]
    pool_type = pool_type.lower()

    edges = np.ones(n + 1, dtype=int) * 1.0 / n
    edges[0] = 0
    edges = np.round(np.cumsum(edges) * m).astype(int)
    for j in range(n):
        if pool_type == 'mean':
            arr[j][...] = x[edges[j]:edges[j + 1], :].mean(axis=0)
        elif pool_type == 'max':
            arr[j][...] = x[edges[j]:edges[j + 1], :].max(axis=0)
        else:
            raise ValueError('Unknown pooling type {}'.format(pool_type))

        if norm:
            feat_norm = np.sqrt((arr[j] ** 2).sum())
            if feat_norm == 0:
                feat_norm = 1.0
            arr[j] /= feat_norm

    concat_feat = np.hstack(arr)
    if unit:
        return concat_feat / n
    return concat_feat
