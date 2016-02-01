import numpy as np

from utils import sampling_with_uniform_groups
from utils import segment_intersection

RATIO_INTERVALS = [0, 0.05, 0.15, 0.5, np.inf]


def generate_segments(t_size, l_size, annotations, cov_edges=RATIO_INTERVALS,
                      i_thr=0.5, rng_seed=None):
    """Sample segments from a video

    Parameters
    ----------
    t_size : int
        Size of the temporal window
    l_size : int
        Size of the video
    annotations : ndarray
        2-dim array with annotations of video, format: [m x 2:=[init, end]]
    cov_edges : ndarray or list
        1-dim array with intervals for discretize coverage
    i_thr : float, optional
        Threshold over intersection to consider that action appears inside a
        segment
    rng_seed : int, optional

    Outputs
    -------
    segments : ndarray
    2-dim array of selected segments for a video, format: [n x 2:=[init, end]]
    new_annotations : list
        container for annotations inside the segment. length's list equal to n

    """
    rng = np.random.RandomState()
    if isinstance(rng_seed, int):
        rng = np.random.RandomState(rng_seed)

    f_init = np.arange(0, l_size - t_size)
    segments = np.stack([f_init, f_init + t_size], axis=-1)
    i_segments, i_ratio = segment_intersection(annotations, segments,
                                               return_ratio_target=True)

    # Coverage computation
    # Note: summing i_ratio of segments may yield values greater that 1.
    idx_mask = i_ratio >= i_thr
    i_ratio[~idx_mask] = 0  # 0 coverage for incomplete actions and empty seg
    cov_ratio_per_segment = i_ratio.sum(axis=0)

    idx_samples = sampling_with_uniform_groups(
        cov_ratio_per_segment, cov_edges, strict=False, rng=rng)
    idx_samples = rng.permutation(idx_samples)

    # Should valif i_segments have intersection >=0.5?
    new_annotations = [None] * len(idx_samples)
    for i, v in enumerate(idx_samples):
        new_annotations[i] = i_segments[idx_mask[:, v], v, :]
    return segments[idx_samples, :], new_annotations
