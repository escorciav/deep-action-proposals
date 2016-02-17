#!/bin/bash/env python
"""

PCA done via matrix multiplication out-of-core. It is here just to be
informative i.e. hostile and full of dependencies parsing of inputs.

"""
import time

import h5py
import hickle as hkl
import numpy as np

THUMOS14_VAL = 'data/thumos14/c3d/val_c3d_temporal.hdf5'


def main(h5file=THUMOS14_VAL, t_size=16, t_stride=8, feat_dim=4096,
         source='c3d_features', log_loop=500000):
    print time.ctime(), 'start: loading hdf5'
    fid = h5py.File(h5file, 'r')
    print time.ctime(), 'finish: loading hdf5'

    # Compute mean
    print time.ctime(), 'start: compute mean'
    x_mean, n = np.zeros((1, feat_dim), dtype=np.float32), 0
    for i, v in fid.iteritems():
        feat = v[source][:]
        n += feat.shape[0]
        x_mean += feat.sum(axis=0)
    x_mean /= n
    print time.ctime(), 'finish: compute mean'

    # Compute A.T A
    print time.ctime(), 'start: out-of-core matrix multiplication'
    j, n_videos = 0, len(fid.keys())
    ATA = np.zeros((feat_dim, feat_dim), dtype=np.float32)
    for i, v in fid.iteritems():
        feat = v[source][:]
        feat_ = feat - x_mean
        ATA += np.dot(feat_.T, feat_)
        j += 1

        if j % log_loop == 0:
            print time.ctime(), 'Iteration {}/{}'.format(j, n_videos)
    print time.ctime(), 'finish: out-of-core matrix multiplication'

    # SVD
    print time.ctime(), 'start: SVD in memory'
    U, S, _ = np.linalg.svd(ATA)
    print time.ctime(), 'finish: SVD in memory'

    print time.ctime(), 'serializing ...'
    hkl.dump({'x_mean': x_mean, 'U': U, 'S': S}, 'pca_val_annot_thumos14.hkl')

if __name__ == '__main__':
    main()
