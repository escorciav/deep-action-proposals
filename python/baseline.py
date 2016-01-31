import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def temp_annot_transf(X, mthd='c2b'):
    """Transform temporal annotations

    Parameters
    ----------
    X : ndarray
        [n x 2] array with temporal annotations
    mthd : str
        Type of conversion:
        'c2b': transform [central-frame, duration] onto [f-init, f-end]

    Outputs
    -------
    Y : ndarray
        [n x 2] array with transformed temporal annotations

    """
    if mthd == 'c2b':
        Xinit = np.round(X[:, 0] - 0.5*X[:, 1])
        return np.stack([Xinit, Xinit + X[:, 1]], axis=-1)


def proposals_per_video(X, n_proposals=None, n_videos=None):
    """
    Reshape ndarray of dim 2 into ndarray of 3 dim where the first dimension
    access all temporal annotations of the i-th video

    Parameters
    ----------
    X : ndarray
    n_proposals : int, optional
    n_videos : int, optional

    Outputs
    -------
    Y : ndarray

    """
    if X.ndim != 2:
        raise ValueError('Array with incorrent number of dimensions')
    if n_proposals is None and n_videos is None:
        raise ValueError('Missing extra information to arrange proposals')
    if n_videos is None:
        n_videos = X.shape[0] / n_proposals
    if n_proposals is None:
        n_proposals = X.shape[0] / n_videos
    return X.reshape((n_videos, n_proposals, X.shape[1]))


class BaselineData(object):
    fields = ['video-name', 'f-init', 'n-frames',
              'video-duration', 'label-idx']

    def __init__(self, df):
        self.data = df
        if df.shape[1] != len(self.fields):
            ValueError('Inconsistent data format')
        self.data.columns = self.fields

    @classmethod
    def fromcsv(cls, filename):
        """Get DataFrame from CSV

        Parameters
        ----------
        filename : str
            Fullpath of CSV-file. It must include header.

        """
        if not os.path.isfile(filename):
            raise IOError('Unknown file {}'.format(filename))

        df = pd.read_csv(filename, sep=' ')
        df_selected = pd.concat([df.loc[:, cls.fields[0]],
                                 df.loc[:, cls.fields[1]],
                                 df.loc[:, cls.fields[2]],
                                 df.loc[:, cls.fields[3]],
                                 df.loc[:, cls.fields[4]]],
                                axis=1, ignore_index=True)
        return cls(df_selected)

    def get_temporal_loc(self):
        """
        Return ndarray with temporal localization of actions in format
        [norm-central-frame, norm-duration]
        """
        norm_duration = (1.0 * self.data.loc[:, self.fields[2]] /
                         self.data.loc[:, self.fields[3]])
        norm_f_center = ((-1.0 + self.data.loc[:, self.fields[1]]) /
                         self.data.loc[:, self.fields[3]]) + norm_duration
        return np.array(pd.concat([norm_f_center, norm_duration], axis=1))


class TempPriorsNoScale(object):
    def __init__(self, n_prop=200):
        self.n_prop = n_prop
        self.model = Pipeline([('scaling', StandardScaler()),
                               ('kmeans', KMeans(n_prop))])

    def fit(self, X):
        """KMeans over temporal locations features
        """
        self.model.fit(X)
        norm_priors = self.model.steps[1][1].cluster_centers_
        mu = self.model.steps[0][1].mean_
        sigma = self.model.steps[0][1].scale_
        self.priors = norm_priors * sigma + mu

    def proposals(self, X, return_index=False):
        """Retrieve proposals for a video based on its duration

        Parameters
        ----------
        X : ndarray
            m x 1 array with video duration

        Outputs
        -------
        Y : ndarray
            m * n_prop x 2 array with temporal proposals

        """
        Y = np.kron(np.expand_dims(X, 1), self.priors)
        idx = np.repeat(np.arange(X.shape[0]), self.n_prop)
        if return_index:
            return Y, idx
        return Y
