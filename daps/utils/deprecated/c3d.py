import glob
import os

import natsort
import numpy as np

import h5py
import hickle as hkl

from daps.utils.c3d import c3d_read_feature
from daps.utils.pooling import pyramid1d


def c3d_stack_feature(dirname, files=None, layer='.fc7-1', savefile=None,
                      pool_type=None):
    """Read C3D features from disk and stack them as ndarray

    Parameters
    ----------
    dirname : str.
        Fullpath of dirname with C3D-output.
    files : list, optional.
        List of basename files without extension to read. By default, stack all
        the features associated with the layer of interest.
    layer : str, optional.
        Layer of interest to read.
    save : str, optional.
        Dump features in one single file.
    pool_type : str, optional.
        Global pooling strategy over a bunch of features. Choices are limited
        'mean', 'max', 'pyr:l,mean/max'.

    Outputs
    -------
    arr : ndarray
        2-dim array of shape [m x d]. d:= is the dimensionality of the feature
        space. m := number of features stacked. If pooling is applied, m = 1.

    Note: It just stacks several flatten 1-dim array read by c3d_read_feature.

    """
    if not os.path.exists(dirname):
        raise IOError('Unexistent folder {}'.format(dirname))

    # Get files to read
    if files is not None and isinstance(files, list):
        c3d_files = [None] * len(files)
        for i, v in enumerate(files):
            c3d_files[i] = os.path.join(dirname, str(v) + layer)
    else:
        c3d_files = glob.glob(os.path.join(dirname, '*' + layer))

    sorted_files = natsort.natsorted(c3d_files)
    # Initialize ndarray
    s, data = c3d_read_feature(sorted_files[0])
    arr = np.empty((len(sorted_files), np.cumprod(s)[-1]))
    arr[0, ...] = data

    # Read most of the features
    for i, v in enumerate(sorted_files[1::]):
        _, data = c3d_read_feature(v)
        arr[i, ...] = data

    # Apply pooling
    if isinstance(pool_type, str):
        pool_type = pool_type.lower()
        if pool_type == 'mean':
            arr = arr.mean(axis=0)
        elif pool_type == 'max':
            arr = arr.max(axis=0)
        elif 'pyr' in pool_type:
            level, pool_type = pool_type.split(':')[1].split(',')
            arr = pyramid1d(arr, int(level), pool_type)
        else:
            raise ValueError('Unknown pool_type: ' + pool_type)
        arr = np.expand_dims(arr, axis=0)

    # Save if required
    if isinstance(savefile, str):
        if len(os.path.splitext(savefile)[1]) <= 0:
            savefile += '.hkl'
        hkl.dump(arr, savefile, mode='w', compression='gzip',
                 compression_opts=9)
    return arr


def c3d_batch_feature_stacking(df, dirname, t_size=16, t_stride=8,
                               feat_size=4096, savedir=None, stack_prm=None,
                               persistent=None, h5mode='x', h5prm=None):
    """Stack C3D feature for a bunch of segments

    Parameters
    ----------
    df : DataFrame.
        Table data with (at  least) the following column names:
        ['video-name', 'f-init', 'duration'].
    dirname : str.
        Fullpath of root folder with features.
    t_size : int, optional.
        Size of temporal receptive field C3D-model.
    t_stride : int, optional.
        Size of temporal stride btw features.
    savedir : str, optional.
        Fullpath of directory to save results, if required.
    stack_prm : dict.
        Parameters for c3d_stack_feature function.
    persistent : str, optional
        save stack of feature-arrays as HDF5. Use it when your memory is scarce
        wrt to the number of rows on df and feature size.
    h5mode : str, optional.
        Mode to open hdf5 file.
    h5prm : dict, optional.
        dict with parameter for HDF5 saved on persistent mode.

    Outputs
    -------
    arr : ndarray.
        3-dim array of shape [df.shape[0], m, d] where d := dimensionality of
        the feature space and m is df.loc[i, 'duration']

    Notes
    -----
    It assumes that segments in df have the same length

    """
    n, T = df.shape[0], df.loc[:, 'duration'].min()
    Te = (T - t_size) / t_stride + 1
    if stack_prm is None:
        stack_prm = {}
    if h5prm is None:
        h5prm = dict(chunks=True, compression='lzf')

    def wrapper_c3d_stacking(video_name, f_init, duration, dirname=dirname,
                             T=t_size, s=t_stride, savedir=savedir,
                             stack_prm=stack_prm):
        # Extra 1 comes from zero-indexing
        frames_of_interest = range(f_init, f_init + duration - T + 1, s)
        dirname_video = os.path.join(dirname, video_name)
        args = (dirname_video, frames_of_interest)

        if isinstance(savedir, str):
            outfile = os.path.join(savedir, video_name + '_' + str(f_init))
            tmp = c3d_stack_feature(*args, savefile=outfile, **stack_prm)
        else:
            tmp = c3d_stack_feature(*args, **stack_prm)

        return tmp

    if persistent:
        f = h5py.File(persistent, h5mode)
        arr = f.create_dataset('segment_features', (n, Te, feat_size),
                               dtype='float32', **h5prm)
        arr.dims[0].label = 'batch'
        arr.dims[1].label = 't_step'
        arr.dims[2].label = 'feature'
    else:
        arr = np.empty((n, T, feat_size), dtype=np.float32)

    for i, v in enumerate(df.index):
        arr[i, ...] = wrapper_c3d_stacking(*df.loc[v, ['video-name', 'f-init',
                                                       'duration']])

    if persistent:
        f.close()
    else:
        return arr
