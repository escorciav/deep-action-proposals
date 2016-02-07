import os
import warnings

import hickle as hkl
import numpy as np
import pandas as pd

from baseline import TempPriorsNoScale
from utils import sampling_with_uniform_groups
from utils import segment_format, segment_unit_scaling
from utils import segment_intersection, segment_iou

RATIO_INTERVALS = [0, 0.05, 0.15, 0.5, np.inf]
REQ_INFO_CP = ['video-name', 'f-init', 'n-frames', 'video-frames']


def wrapper_unit_scaling(x, T, s_ref, n_gt, *args, **kwargs):
    """Normalize segments to unit-length and use center-duration format
    """
    xc = segment_format(x, 'b2c')
    init_ref = np.repeat(s_ref[:, 0], n_gt)
    return segment_unit_scaling(xc, T, init_ref)


def compute_priors(df, T, K=200, iou_thr=0.5, norm_fcn=wrapper_unit_scaling):
    """Clustering of ground truth locations

    Parameters
    ----------
    X : DataFrame
        pandas table with annotations of the dataset. It must include the
        following columns data_generation.REQ_INFO_CP
    T : int
        canonical temporal size of evaluation window
    K : int, optional
        number of priors
    iou_thr : float
        IOU threshold to consider that an annotation match with a prior
    norm_fcn : function
        Function to apply over ndarray [m x 2] of segments with
        format :=[f-init, f-end] before computing priors.

    Outputs
    -------
    priors : ndarray
        2-dim array of priors discovered. The first dimension iterates over the
        different priors.
    new_df : DataFrame
        Table with information about instances to use in training

    """
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise ValueError('df argument must be a pd.DataFrame')
    if not set(REQ_INFO_CP).issubset(df.columns.tolist()):
        msg = 'df must include these column names: {}'.format(REQ_INFO_CP)
        raise ValueError(msg)
    if iou_thr > 1 or iou_thr < 0:
        raise ValueError('Invalid value of IOU')

    # Loop over videos
    videos = df['video-name'].unique()
    L = np.empty(videos.size, dtype=int)
    segment_lst, n_seg = [None] * videos.size, np.empty(videos.size, dtype=int)
    mapped_gt_lst, n_gt_lst = [None] * videos.size, [None] * videos.size
    for i, v in enumerate(videos):
        idx = df['video-name'] == v
        L[i] = df.loc[idx, 'video-frames'].mean()
        gtruth_c = df.loc[idx, ['f-init', 'n-frames']]
        gtruth_b = segment_format(np.array(gtruth_c), 'c2b')
        segment_lst[i], gt_list_i, n_gt_lst[i] = generate_segments(
            T, L[i], gtruth_b)
        n_seg[i] = segment_lst[i].shape[0]
        mapped_gt_lst[i] = np.vstack(gt_list_i)

    # Standardize mapped annotations into a common reference + Normalization
    segments = np.vstack(segment_lst)
    mapped_gt = np.vstack(mapped_gt_lst)
    n_gt = np.hstack(n_gt_lst)
    X = norm_fcn(mapped_gt, T, segments, n_gt)

    # Clustering
    model = TempPriorsNoScale(K)
    model.fit(X)
    priors = model.priors

    # Matching
    score = np.zeros((segments.shape[0], priors.shape[0]))
    for i, v in enumerate(segment_lst):
        # Scale priors and use boundary format
        mapped_priors_b = segment_format(priors * T, 'c2b')
        s_ref = np.expand_dims(np.repeat(v[:, 0], n_gt_lst[i]), 1)
        # Reference mapped gt on [0 - T] interval
        mapped_gt_i_ref = mapped_gt_lst[i] - s_ref

        if mapped_gt_i_ref.size == 0:
            continue
        if (mapped_gt_i_ref[:, 0] < 0).sum() > 0:
            msg = ('Initial frame must be greater that zero. Running at your '
                   'own risk. Debug is needed.')
            warnings.warn(msg)

        # IOU computation
        iou = segment_iou(mapped_priors_b, mapped_gt_i_ref)
        score[i, :] = iou.max(axis=1) > iou_thr

    # Build DataFrame
    col_triads = ['c_{}'.format(i) for i in range(K)]
    new_df = pd.concat([pd.DataFrame({'video-name': videos.repeat(n_seg),
                                      'f-init': segments[:, 0],
                                      'duration': np.repeat(T,
                                                            segments.shape[0]),
                                      'video-frames': np.repeat(L, n_seg)}),
                        pd.DataFrame(score, columns=col_triads)], axis=1)
    return priors, new_df


def dump_files(filename, priors=None, df=None, conf=False):
    """Dump files used to train the model and feature extraction

    Parameters
    ----------
    filename : str
        Fullpath of prefix name for helper-files
    priors : ndarray, optional
        2-dim array of location priors
    df : DataFrame, optional
        Table returned by compute_priors
    conf : bool, optional
        Save a file with confidence of each prior on every segment

    Note: Files should be parse together to get insightful understanding of
    the information because no further indexing is included in each file.

    """
    filefmt = filename + '_{}.{}'
    # HDF5 with priors
    if priors is not None:
        hkl.dump(priors, filefmt.format('priors', 'hkl'), mode='w',
                 compression='gzip', compression_opts=9)

    # List of videos ready for C3D feature extractor wrapper
    if df is not None:
        df.rename(columns={'video-frames': 'num-frame',
                           'f-init': 'i-frame'}, inplace=True)
        lst = ['video-name', 'num-frame', 'i-frame', 'duration']
        df[lst].to_csv(filefmt.format('ref', 'lst'), sep=' ', index=False)
        # Rename columns again to avoid modify df without increase memory
        df.rename(columns={'num-frame': 'video-frames',
                           'i-frame': 'f-init'}, inplace=True)

    # HDF5 with confidences
    if conf and df is not None:
        lst = ['c_{}'.format(i) for i in range(df.columns.size - 4)]
        hkl.dump(np.array(df.loc[:, lst]), filefmt.format('conf', 'hkl'),
                mode='w', compression='gzip', compression_opts=9)


def generate_segments(t_size, l_size, annotations, cov_edges=RATIO_INTERVALS,
                      i_thr=0.5, rng_seed=None, return_annot=True):
    """Sample segments from a video

    Parameters
    ----------
    t_size : int
        Size of the temporal window.
    l_size : int
        Size of the video.
    annotations : ndarray
        2-dim array with annotations of video, format: [m x 2:=[init, end]].
    cov_edges : ndarray or list
        1-dim array with intervals for discretize coverage.
    i_thr : float, optional
        Threshold over intersection to consider that action appears inside a
        segment.
    rng_seed : int, optional
    return_annot : bool, optional
        Return two extra outputs (new_annotations and n_annotations).

    Outputs
    -------
    segments : ndarray
        2-dim array of selected segments for a video,
        format: [n x 2:=[init, end]].
    new_annotations : list
        container for annotations inside the segment. length's list equal to n.
    n_annotations : ndarray
        1-dim array of size n with number of annotations inside ith-segment.

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
    if return_annot:
        new_annotations = [None] * len(idx_samples)
        n_annotations = np.zeros(len(idx_samples), dtype=int)
        for i, v in enumerate(idx_samples):
            new_annotations[i] = i_segments[idx_mask[:, v], v, :]
            n_annotations[i] = new_annotations[i].shape[0]
        return segments[idx_samples, :], new_annotations, n_annotations

    return segments[idx_samples, :]


def load_files(priors_file=None, ref_file=None, conf_file=None):
    """Load files output by dump_files

    Parameters
    ----------
    priors_file : str
    ref_file : str
    conf_file : str

    Outputs
    -------
    priors: ndarray
    df : DataFrame
    conf : ndarray

    """
    def check_file_existence(filename, suffix):
        if not os.path.exists(filename):
            if os.path.exists(filename + suffix):
                filename += suffix
            else:
                raise IOError('Unknown file ' + filename)
        return filename

    if isinstance(priors_file, str):
        priors_file = check_file_existence(priors_file, '_priors.hkl')
        priors = hkl.load(priors_file)
    else:
        priors = None

    if isinstance(ref_file, str):
        ref_file = check_file_existence(ref_file, '_ref.lst')
        df = pd.read_csv(ref_file, sep=' ')
        # Rename columns
        df.rename(columns={'num-frame': 'video-frames',
                           'i-frame': 'f-init'}, inplace=True)
    else:
        df = None

    if isinstance(conf_file, str):
        conf_file = check_file_existence(conf_file, '_conf.hkl')
        conf = hkl.load(conf_file)
        if isinstance(df, pd.DataFrame):
            df_conf = pd.DataFrame(conf)
            df_conf.columns = ['c_{}'.format(i)
                               for i in range(0, conf.shape[1])]
            df = pd.concat([df, df_conf], axis=1)
    else:
        conf = None
    return priors, df, conf
