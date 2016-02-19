import os
import warnings

import hickle as hkl
import numpy as np
import pandas as pd

from baseline import TempPriorsNoScale
from utils import sampling_with_uniform_groups
from utils import segment_format, segment_unit_scaling
from utils import segment_intersection, segment_iou

RATIO_INTERVALS = [0, 0.05, 0.15, 0.4, np.inf]
REQ_INFO_CP = ['video-name', 'f-init', 'n-frames', 'video-frames']


def wrapper_unit_scaling(x, T, s_ref, n_gt, *args, **kwargs):
    """Normalize segments to unit-length and use center-duration format
    """
    xc = segment_format(x, 'b2c')
    init_ref = np.repeat(s_ref[:, 0], n_gt)
    return segment_unit_scaling(xc, T, init_ref)


def compute_priors(df, T, K=200, iou_thr=0.5, norm_fcn=wrapper_unit_scaling,
                   i_thr=1.0, rng_seed=None):
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
    i_thr : float
        ratio [0, 1] to include an annotation inside a segment.
    rng_seed : int
        Seed for random number generator

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
        gtruth_b = segment_format(np.array(gtruth_c), 'd2b')
        segment_lst[i], gt_list_i, n_gt_lst[i] = generate_segments(
            T, L[i], gtruth_b, method='iou', rng_seed=rng_seed, i_thr=i_thr)
        n_seg[i] = segment_lst[i].shape[0]
        mapped_gt_lst[i] = np.vstack(gt_list_i)

    # Standardize mapped annotations into a common reference + Normalization
    segments = np.vstack(segment_lst)
    mapped_gt = np.vstack(mapped_gt_lst)
    n_gt = np.hstack(n_gt_lst)
    X = norm_fcn(mapped_gt, T, segments, n_gt)

    # Clustering
    model = TempPriorsNoScale(K, rng_seed=rng_seed)
    model.fit(X)
    priors = model.priors

    # Matching
    score = np.empty((segments.shape[0], priors.shape[0]), dtype=int)
    j = 0
    for i, v in enumerate(segment_lst):
        # Scale priors and use boundary format
        mapped_priors_b = segment_format(priors * T, 'c2b')
        s_ref = np.expand_dims(np.repeat(v[:, 0], n_gt_lst[i]), 1)

        # Reference mapped gt on [0 - T] interval
        if mapped_gt_lst[i].size == 0:
            continue
        mapped_gt_i_ref = mapped_gt_lst[i] - s_ref
        if (mapped_gt_i_ref[:, 0] < 0).sum() > 0:
            msg = ('Initial frame must be greater that zero. Running at your '
                   'own risk. Debug is needed.')
            warnings.warn(msg)

        # IOU computation
        iou = segment_iou(mapped_priors_b, mapped_gt_i_ref)

        # Map IOU of priors for each segment
        idx = [0] + np.cumsum(n_gt_lst[i]).tolist()
        max_iou = np.vstack(map(lambda u, v: np.zeros(K, dtype=int)
                                if u == v else iou[:, u:v].max(axis=1),
                                idx[:-1], idx[1::]))
        score[j:j+n_seg[i], :] = max_iou > iou_thr
        j += n_seg[i]

    # Build DataFrame
    col_triads = ['c_{}'.format(i) for i in range(K)]
    new_df = pd.concat([pd.DataFrame({'video-name': videos.repeat(n_seg),
                                      'f-init': segments[:, 0],
                                      'duration': np.repeat(T,
                                                            segments.shape[0]),
                                      'video-frames': np.repeat(L, n_seg)}),
                        pd.DataFrame(score, columns=col_triads)], axis=1)
    return priors, new_df


def compute_priors_over_time(mapped_priors_b, T, l_size, stride=16):
    """Compile priors over time from a given video length.
    Parameters
    ----------
    mapped_priors_b: ndarray
        2-dim array of priors discovered in 'c2b' format [f-init, f-end].
    T: int
        Canonical temporal size of evaluation window.
    l_size : int
        Size of the video.
    stride: int
        Size of the sliding step.

    Outputs
    -------
    priors_t: ndarray
         2-dim array of selected segments for a video,
         format: [n x 2:=[init, end]].
    k_idx: ndarray
         1-dim array of prior indices.
    """
    # Build segment stack.
    nr_priors = mapped_priors_b.shape[0]
    f_init = np.arange(1, l_size - T, stride + 1).repeat(nr_priors)
    priors_t = np.stack([f_init, f_init + np.zeros(f_init.shape[0])], axis=-1)
    k_idx = np.empty(f_init.shape[0])

    # Assign a segment to each prior.
    for i, mp_i in enumerate(mapped_priors_b):
        idx = np.arange(i, priors_t.shape[0], nr_priors)
        priors_t[idx, :] += mp_i
        k_idx[idx] = i
    return priors_t, k_idx


def evaluate_priors(df, priors, T, stride=16, iou_thr=0.5,
                    return_recall=False):
    """
    Parameters
    ----------
    df: DataFrame
        Pandas table with annotations of the dataset. It must include the
        following columns data_generation.REQ_INFO_CP
    priors: ndarray
        2-dim array of priors discovered. The first dimension iterates over the
        different priors.
    T: int
        Canonical temporal size of evaluation window.
    stride: int, optional
        Size of the sliding step.
    iou_thr : float, optional
        IOU threshold to consider that an annotation match with a prior.
    return_recall: bool, optional
        Return one extra output (recall, computed at given iou_thr).

    Outputs
    -------
    eval_df: DataFrame
        Table with information about each annotation and its matched prior.
    recall: float
        Recall at given iou threshold.
    """
    # Sanitize input.
    mapped_priors_b = segment_format(priors * T, 'c2b').clip(1, T)
    mapped_priors_b = np.array(mapped_priors_b).astype(np.int)

    # Iterate over each instance.
    best_iou, v_pointer = np.empty(df['video-name'].size), 0
    best_priors_t = np.empty((df['video-name'].size, 2))
    best_priors_index = np.empty(df['video-name'].size)
    for i, sgm_i in df.iterrows():
        # Parsing ground-truth.
        L = sgm_i['video-frames']
        gtruth_c = np.empty((1, 2))
        gtruth_c[0, :] = np.stack([sgm_i['f-init'], sgm_i['n-frames']],
                                  axis=-1)
        gtruth_b = segment_format(gtruth_c, 'd2b')

        # Slide priors over time.
        priors_t, k_idx = compute_priors_over_time(mapped_priors_b, T,
                                                   L, stride)

        # Not found priors for this video.
        if priors_t.shape[0] == 0:
            best_iou[v_pointer] = 0.0
            best_priors_t[v_pointer, :] = np.array([[np.nan, np.nan]])
            best_priors_index[v_pointer] = np.array([np.nan])
            v_pointer += 1
            continue

        # Compute iou and keep the best one for each ground-truth instance.
        iou = segment_iou(gtruth_b, priors_t)
        max_idx = iou.argmax(axis=1)
        best_iou[v_pointer] = iou.flatten()[max_idx]
        best_priors_t[v_pointer, :] = priors_t[max_idx, :]
        best_priors_index[v_pointer] = k_idx[max_idx]
        v_pointer += 1

    # Build DataFrame.
    s_init = best_priors_t[:, 0]
    n_frames = best_priors_t[:, 1] - best_priors_t[:, 0] + 1
    eval_df = pd.concat([df, pd.DataFrame({'priors-f-init': s_init,
                                           'priors-n-frames': n_frames,
                                           'k-idx': best_priors_index,
                                           'iou': best_iou})], axis=1)
    if return_recall:
        n_annotations = eval_df.shape[0]
        recall = (eval_df['iou'] >= iou_thr).sum().astype(float)/n_annotations
        return eval_df, recall
    return eval_df


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
        hkl.dump(priors.astype(np.float32),
                 filefmt.format('priors', 'hkl'), mode='w',
                 compression='gzip', compression_opts=1)

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
        hkl.dump(np.array(df.loc[:, lst]).astype(np.int32),
                 filefmt.format('conf', 'hkl'),
                 mode='w', compression='gzip', compression_opts=1)


def generate_segments(t_size, l_size, annotations, cov_edges=RATIO_INTERVALS,
                      i_thr=0.5, rng_seed=None, method=None,
                      strict_uniform=False, return_annot=True):
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
    method : (None, str)
        Flag to select type of 'coverage': (None or 'iou')
    strict_uniform: bool
        If true the samples are selected with strict uniform distribution.
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

    f_init = np.arange(1, l_size - t_size)
    segments = np.stack([f_init, f_init + t_size - 1], axis=-1)
    i_segments, i_ratio = segment_intersection(annotations, segments,
                                               return_ratio_target=True)

    idx_mask = i_ratio >= i_thr
    if isinstance(method, str):
        iou_ratio = segment_iou(annotations, segments)
        cov_ratio_per_segment = iou_ratio.sum(axis=0)
    else:
        # Coverage computation
        # Note: summing i_ratio of segments may yield values greater that 1.
        i_ratio[~idx_mask] = 0  # For incomplete actions and empty seg.
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
