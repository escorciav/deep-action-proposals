import numpy as np


def format(X, mthd='c2b', T=None, init=None):
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


def intersection(target_segments, test_segments, return_ratio_target=False):
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


def iou(target_segments, test_segments):
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


def unit_scaling(X, T, init=None, copy=False):
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
