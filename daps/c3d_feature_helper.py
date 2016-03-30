import h5py
import numpy as np


class Feature(object):
    def __init__(self, filename, feat_id='c3d_features',
                 t_size=16, t_stride=8, pool_type='mean'):
        """
        Parameters
        ----------
        filename : str.
            Full path to the hdf5 file.
        feat_id : str, optional.
            Dataset identifier.
        t_size : int, optional.
            Size of temporal receptive field C3D-model.
        t_stride : int, optional.
            Size of temporal stride between features.
        pool_type : str, optional.
            Global pooling strategy over a bunch of features.
            'mean', 'max', 'pyr-2-mean/max', 'concat-2-mean/max'
        """
        self.filename = filename
        with h5py.File(self.filename, 'r') as fobj:
            if not fobj:
                raise ValueError('Invalid type of file.')
        self.feat_id = feat_id
        self.fobj = None
        self.t_size = t_size
        self.t_stride = t_stride
        self.pool_type = pool_type

    def open_instance(self):
        """Open file and keep it open till a close call.
        """
        self.fobj = h5py.File(self.filename, 'r')

    def close_instance(self):
        """Close existing h5py object instance.
        """
        if not self.fobj:
            raise ValueError('The object instance is not open.')
        self.fobj.close()
        self.fobj = None

    def read_feat(self, video_name, f_init=None, duration=None,
                  return_reshaped=True):
        """Read C3D features and stack them into memory.

        Parameters
        ----------
        video-name : str.
            Video identifier.
        f_init : int, optional.
            Initial frame index. By default the feature is
            sliced from frame 1.
        duration : int, optional.
            Duration in term of number of frames. By default
            it is set till the last feature.
        return_reshaped : bool.
            Return stack of features reshaped when pooling is applied.
        """
        if not self.fobj:
            raise ValueError('The object instance is not open.')
        T = self.t_size
        s = self.t_stride
        if f_init and duration:
            frames_of_interest = range(f_init, f_init + duration - T + 1, s)
            feat = self.fobj[video_name][self.feat_id][frames_of_interest, :]
        elif f_init and (not duration):
            feat = self.fobj[video_name][self.feat_id][f_init:-T+1:s, :]
        elif (not f_init) and duration:
            feat = self.fobj[video_name][self.feat_id][:duration-T+1:s, :]
        else:
            feat = self.fobj[video_name][self.feat_id][:-T+1:s, :]
        pooled_feat = self._feature_pooling(feat)

        if not return_reshaped:
            feat_dim = feat.shape[1]
            pooled_feat = pooled_feat.reshape((-1, feat_dim))
            if not pooled_feat.flags['C_CONTIGUOUS']:
                return np.ascontigousarray(pooled_feat)
        return pooled_feat

    def read_feat_batch_from_video(self, video_name, f_init_array,
                                   duration=256, return_reshaped=True):
        """Read C3D feature batch from a video. The slicing
           is operated in memory.

        Parameters
        ----------
        video-name : str.
            Video identifier.
        f_init_array : 1darray.
            Contains list of initial frames.
        duration : int.
            Segment size.
        return_reshaped : bool.
            Return stack of features reshaped when pooling is applied.
        """
        if not self.fobj:
            raise ValueError('The object instance is not open.')
        # Sanitize.
        f_init_array = f_init_array.astype(int)
        # Load all features associated to video-name.
        raw_feat_stack = self.fobj[video_name][self.feat_id].value
        t_size = self.t_size
        s = self.t_stride
        n_segments = f_init_array.shape[0]

        # Set feat stack size.
        m = 1
        d = raw_feat_stack.shape[1]
        if 'pyr' in self.pool_type:
            _, levels, pool_type = self.pool_type.split('-')
            d *= 2**(int(levels) + 1) - 1
        elif 'concat' in self.pool_type:
            _, levels, pool_type = self.pool_type.split('-')
            d *= int(levels)
        elif not self.pool_type:
            m *= (duration - t_size)/s + 1
        feat_stack = np.empty((n_segments, int(m), int(d)))

        # Iterate over each segment.
        for i, f_init in enumerate(f_init_array):
            frames_of_interest = range(f_init,
                                       f_init + duration - t_size + 1, s)
            feat_stack[i, ...] = self._feature_pooling(
                raw_feat_stack[frames_of_interest, :])

        if return_reshaped and self.pool_type:
            feat_stack = feat_stack.reshape(feat_stack.shape[0],
                                            feat_stack.shape[2])
        return feat_stack

    def _feature_pooling(self, x):
        """Compute pooling of a feature vector.

        Parameters
        ----------
        x : ndarray.
            [m x d] array of features.m is the number of features and
            d is the dimensionality of the feature space.
        """
        if x.ndim != 2:
            raise ValueError('Invalid input ndarray. Input must be [mxd].')

        if not self.pool_type:
            return x

        if self.pool_type == 'mean':
            return x.mean(axis=0)
        elif self.pool_type == 'max':
            return x.max(axis=0)
        elif 'pyr' in self.pool_type:
            _, level, pool_type = self.pool_type.split('-')
            return pyramid1d(x, int(level), pool_type)
        elif 'concat' in self.pool_type:
            _, level, pool_type = self.pool_type.split('-')
            return concat1d(x, int(level), pool_type)


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
