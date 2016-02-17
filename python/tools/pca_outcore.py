#!/bin/bash/env python
"""

PCA done via matrix multiplication out-of-core. It is here just to be
informative i.e. hostile and full of dependencies parsing of inputs.

"""
import os
import time

import hickle as hkl
import numpy as np

from utils import c3d_stack_feature
from data_generation import load_files

C3D_DIR = 'data/thumos14/c3d/val/'
REFFILE = 'data/experiments/thumos14/a01/train_ref.lst'


def read_feature(frame, dirname=C3D_DIR):
    return c3d_stack_feature(dirname, [frame]).astype(np.float32)


def read_reffile(filename):
    """Load reference lst-file dump by data_generation.dump_files
    """
    _, df, _ = load_files(ref_file=filename)
    return df


def main(ref_file, t_size=16, t_stride=8, feat_dim=4096, dirname=C3D_DIR,
         log_loop=10000):
    print time.ctime(), 'start: input parsing'
    df = read_reffile(ref_file)

    # create dataframe with df, t
    n = df.shape[0]
    T = df.loc[:, 'duration'].min()
    n_feat = (T - t_size) / t_stride + 1
    frames = np.empty((n * n_feat), dtype=int)
    for i, v in df.iterrows():
        t_init = v['f-init']
        frames[i*n_feat:(i+1)*n_feat] = np.arange(
            t_init, t_init + T - 1, t_stride)[:n_feat]
    video_name = df['video-name'].repeat(n_feat)
    print time.ctime(), 'finish: input parsing'

    # Compute mean
    print time.ctime(), 'start: compute mean'
    x_mean = np.zeros((1, feat_dim), dtype=np.float32)
    for i in xrange(frames.size):
        v = video_name.iloc[i]
        filename = os.path.join(dirname, v)
        x_mean += read_feature(frames[i], filename)
    x_mean /= n
    print time.ctime(), 'finish: compute mean'

    # Compute A.T A
    print time.ctime(), 'start: out-of-core matrix multiplication'
    ATA = np.zeros((feat_dim, feat_dim), dtype=np.float32)
    for i in xrange(frames.size):
        v = video_name.iloc[i]
        filename = os.path.join(dirname, v)
        xi = read_feature(frames[i], filename)
        xi_ = xi - x_mean
        ATA += np.dot(xi_.T, xi_)

        if i % log_loop == 0:
            print time.ctime(), 'Iteration {}/{}'.format(i, n * n_feat)
    print time.ctime(), 'finish: out-of-core matrix multiplication'

    # SVD
    print time.ctime(), 'start: SVD in memory'
    U, S, V = np.linalg.svd(ATA)
    print time.ctime(), 'finish: SVD in memory'

    print time.ctime(), 'serializing ...'
    hkl.dump({'x_mean': x_mean, 'U': U, 'S': S, 'V': V}, 'pca.hkl')


if __name__ == '__main__':
    main(REFFILE)
