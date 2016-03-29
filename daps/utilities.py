import json
import os

import numpy as np


# General utilities
def balance_labels(Y):
    """Compute weights to balance distribution btw 1/0

    Parameters
    ----------
    Y : ndarray
        Binary Label matrix or vector {0, 1}.

    Outputs
    -------
    w_pos : float
        Weight for samples labeled as 1
    w_neg : float
        Weight for samples labeled as 0

    """
    w_pos = Y.sum() * 1.0
    w_neg = Y.size * 1.0 - w_pos

    if w_pos > w_neg:
        w_neg, w_pos = 1.0, w_neg / w_pos
    elif w_pos < w_neg:
        w_neg, w_pos = w_pos / w_neg, 1.0
    else:
        w_pos = w_neg = 1.0
    return w_pos, w_neg


def feature_1dpyramid(x, levels=0, pool_type='mean', norm=True, unit=False):
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


def feature_1dconcat(x, n=8, pool_type='mean', norm=True, unit=False):
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


def idx_of_queries(df, col_name, queries, n_samples=None, rng_seed=None):
    """Return indexes of several queries on a DataFrame

    Parameters
    ----------
    df : DataFrame
    col_name : int, str
    queries : list, series, 1-dim ndarray
    n_samples : int
    rng_seed : rng instance or int

    Outputs
    -------
    idx : ndarray
        1-dim array of index over of queries

    """
    idx_lst = [None] * queries.size
    # There should be a pandas way of doing this
    for i, v in enumerate(queries):
        idx_lst[i] = (df[col_name] == v).nonzero()[0]
    idx = np.hstack(idx_lst)

    if n_samples is None:
        return idx
    elif isinstance(n_samples, int):
        if rng_seed is None or isinstance(rng_seed, int):
            rng = np.random.RandomState(rng_seed)
        else:
            rng = rng_seed

        return rng.permutation(idx)[:n_samples]


def uniform_batches(Y, batch_size=1, contiguous=True, return_all=True):
    """Distribute positive labels as uniform as possible among mini batches

    Parameters
    ----------
    Y : ndarray
        Label vector or matrix with samples along the rows.
    batch_size : int, optional
        Size of mini-batches
    contiguous : bool, optional
        Enforce a contiguous array in the output
    return_all : bool, optional
        return all data samples.

    Outputs
    -------
    Yt : ndarray
        Re-organized label vector or matrix
    idx : ndarray
        indexes used to organize data

    Notes
    -----
    This function is not tested on the case that negative labels are scarce.

    """
    if Y.ndim > 1:
        pos_idx = np.where(Y.sum(axis=1) > 0)[0]
    else:
        pos_idx = np.where(Y > 0)[0]

    n = Y.shape[0]
    n_batches = n / batch_size
    neg_idx = np.random.permutation(np.setdiff1d(np.arange(n), pos_idx))

    # Replicate positives if batch size is too small compared to num positives
    if n_batches > pos_idx.size:
        pos_idx = np.tile(pos_idx, np.ceil(n_batches * 1.0 / pos_idx.size))
    # Should not we do something similar with negative?

    pos_groups = np.array_split(pos_idx, n_batches)
    idx_list, j = [None] * n_batches, 0
    for i, v in enumerate(pos_groups):
        m = batch_size - v.size
        idx_list[i] = np.random.permutation(np.hstack([v, neg_idx[j:j + m]]))
        j += m
    if j < neg_idx.size and return_all:
        idx_list.append(neg_idx[j::])
    idx = np.hstack(idx_list)
    if contiguous:
        return np.ascontiguousarray(Y[idx, ...]), idx
    return Y[idx, ...], idx


# Sampling
def sampling_with_uniform_groups(x, bin_edges, strict=True, rng=None):
    """
    Sample values of x such that the distribution on bin-edges is as uniform as
    possible

    Parameters
    ----------
    x : ndarray
        1-dim array to sample
    bin_edges : ndarray
        1-dim array with intervals. See numpy.digitize for more details.
    strict : bool, optional
        If true, every bucket will have the same number of samples.
    rng : numpy.random.RandomState, optional
        pseudo-rnd number generator instance

    Outputs
    -------
    keep_idx : ndarray
        1-dim array of indexes from the elements of x to keep

    """
    if rng is None:
        rng = np.random.RandomState()

    n_bins = len(bin_edges) - 1
    idx = np.digitize(x, bin_edges) - 1
    counts = np.bincount(idx, minlength=n_bins)

    min_samples = counts.min()
    if strict:
        samples_per_bin = np.repeat(min_samples, n_bins)
        # Sample from the same distrib withot matter strict value
        rng.rand(n_bins)
    else:
        samples_per_bin = np.minimum(min_samples+counts.std()*rng.rand(n_bins),
                                     counts)

    keep_idx_list = []
    for i in xrange(n_bins):
        tmp = (idx == i).nonzero()[0]
        keep_idx_list.append(rng.permutation(tmp)[:samples_per_bin[i]])
    keep_idx = np.hstack(keep_idx_list)
    return keep_idx


# Segment utilities
def segment_format(X, mthd='c2b', T=None, init=None):
    """Transform temporal annotations

    Parameters
    ----------
    X : ndarray
        [n x 2] array with temporal annotations
    mthd : str
        Type of conversion:
        'c2b': transform [center, duration] onto [f-init, f-end]
        'b2c': inverse of c2b
        'd2b': transform ['f-init', 'n-frames'] into ['f-init', 'f-end']

    Outputs
    -------
    Y : ndarray
        [n x 2] array with transformed temporal annotations

    """
    if X.ndim != 2:
        msg = 'Incorrect number of dimensions. X.shape = {}'
        ValueError(msg.format(X.shape))

    if mthd == 'c2b':
        Xinit = np.ceil(X[:, 0] - 0.5*X[:, 1])
        Xend = Xinit + X[:, 1] - 1.0
        return np.stack([Xinit, Xend], axis=-1)
    elif mthd == 'b2c':
        Xc = np.round(0.5*(X[:, 0] + X[:, 1]))
        d = X[:, 1] - X[:, 0] + 1.0
        return np.stack([Xc, d], axis=-1)
    elif mthd == 'd2b':
        Xinit = X[:, 0]
        Xend = X[:, 0] + X[:, 1] - 1.0
        return np.stack([Xinit, Xend], axis=-1)


def segment_intersection(target_segments, test_segments,
                         return_ratio_target=False):
    """Compute intersection btw segments

    Parameters
    ----------
    target_segments : ndarray
        2-dim array in format [m x 2:=[init, end]]
    test_segments : ndarray
        2-dim array in format [n x 2:=[init, end]]
    return_ratio_target : bool, optional
        extra ndarray output with ratio btw size of intersection over size of
        target-segments

    Outputs
    -------
    intersect : ndarray
        3-dim array in format [m, n, 2:=[init, end]]
    ratio_target : ndarray
        2-dim array [m x n] with ratio btw size of intersect over size of
        target segment

    Note: It assumes that target-segments are more scarce that test-segments

    """
    if target_segments.ndim != 2 or test_segments.ndim != 2:
        raise ValueError('Dimension of arguments is incorrect')
    m, n = target_segments.shape[0], test_segments.shape[0]
    if return_ratio_target:
        ratio_target = np.zeros((m, n))

    intersect = np.zeros((m, n, 2))
    for i in xrange(m):
        target_size = target_segments[i, 1] - target_segments[i, 0] + 1.0
        tt1 = np.maximum(target_segments[i, 0], test_segments[:, 0])
        tt2 = np.minimum(target_segments[i, 1], test_segments[:, 1])

        intersect[i, :, 0], intersect[i, :, 1] = tt1, tt2
        if return_ratio_target:
            isegs_size = (tt2 - tt1 + 1.0).clip(0)
            ratio_target[i, :] = isegs_size / target_size

    if return_ratio_target:
        return intersect, ratio_target
    return intersect


def segment_iou(target_segments, test_segments):
    """Compute intersection over union btw segments

    Parameters
    ----------
    target_segments : ndarray
        2-dim array in format [m x 2:=[init, end]]
    test_segments : ndarray
        2-dim array in format [n x 2:=[init, end]]

    Outputs
    -------
    iou : ndarray
        2-dim array [m x n] with IOU ratio.

    Note: It assumes that target-segments are more scarce that test-segments

    """
    if target_segments.ndim != 2 or test_segments.ndim != 2:
        raise ValueError('Dimension of arguments is incorrect')

    m, n = target_segments.shape[0], test_segments.shape[0]
    iou = np.empty((m, n))
    for i in xrange(m):
        tt1 = np.maximum(target_segments[i, 0], test_segments[:, 0])
        tt2 = np.minimum(target_segments[i, 1], test_segments[:, 1])

        # Non-negative overlap score
        intersection = (tt2 - tt1 + 1.0).clip(0)
        union = ((test_segments[:, 1] - test_segments[:, 0] + 1) +
                 (target_segments[i, 1] - target_segments[i, 0] + 1) -
                 intersection)
        # Compute overlap as the ratio of the intersection
        # over union of two segments at the frame level.
        iou[i, :] = intersection / union
    return iou


def segment_unit_scaling(X, T, init=None, copy=False):
    """Scale segments onto a unit reference scale [0, 1]

    Parameters
    ----------
    X : ndarray
        [n x 2] array with temporal annotations in [center, duration] format
    T : int, optional
        duration
    init : ndarray
        [n] array with initial value of temporal reference

    Outputs
    -------
    Y : ndarray
        [n x 2] array with transformed temporal annotations

    """
    if X.ndim != 2:
        raise ValueError('Incorrect number of dimension on X')
    Y = X
    if copy:
        Y = X.copy()

    if init is not None:
        if init.size != Y.shape[0]:
            raise ValueError('Incompatible reference, init')
        Y[:, 0] -= init

    Y[:, 0] /= T - 1
    Y[:, 1] /= T
    return Y


# String utilities
def levenshtein_distance(s1, s2):
    """Compute Levenshtein distance btw two strings

    Note
    ----
    Taken from wikibooks.org-wiki-Algorithm_Implementation

    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # j+1 instead of j since previous_row and current_row are one
            # character longer than s2
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


# IO utilities
def dump_json(filename, data, **kwargs):
    """Serialize data as JSON-file

    Parameters
    ----------
    filename : str
    data : list, dict, etc.
        Data to save. Chech json.dump for more details and description.

    """
    with open(filename, 'w') as f:
        json.dump(data, f, **kwargs)
    return None


def file_as_folder(filename):
    """Return a filename ending with os-path-separator

    Parameters
    ----------
    filename : str
        Fullpath filename

    Outputs
    -------
    filename : str
        Fullpath filename ending with os-path-separator

    """
    return os.path.splitext(filename)[0] + os.path.sep
