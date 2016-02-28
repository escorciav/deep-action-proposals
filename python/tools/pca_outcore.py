#!/usr/bin/env python
"""

PCA done via matrix multiplication out-of-core. It is here just to be
informative an argument parser is welcome.

"""
import argparse
import time

import h5py
import hickle as hkl
import numpy as np
from joblib import delayed, Parallel


def main(dsfile, pcafile, t_size=16, t_stride=8, source='c3d_features',
         n_jobs=1):
    print time.ctime(), 'start: loading hdf5'
    fid = h5py.File(dsfile, 'r')
    video_names = fid.keys()
    print time.ctime(), 'finish: loading hdf5'

    def compute_mean(chunk, f=fid, source=source):
        feat_dim = f[chunk[0]][source].shape[1]
        mean_c, n_c = np.zeros((1, feat_dim), dtype=np.float32), 0
        for i in chunk:
            feat_c = f[i][source][:]
            n_c += feat_c.shape[0]
            mean_c += feat_c.sum(axis=0)
        return mean_c, n_c

    print time.ctime(), 'start: compute mean'
    rst = Parallel(n_jobs=n_jobs)(delayed(compute_mean)(video_names[i::n_jobs])
                                  for i in xrange(n_jobs))
    x_mean, n = rst
    for mean_c, n_c in rst[1::]:
        x_mean += mean_c
        n += n_c
    x_mean /= n
    print time.ctime(), 'finish: compute mean'

    def compute_ATA(chunk, f=fid, source=source, mean=x_mean):
        feat_dim = f[chunk[0]][source].shape[1]
        ATA_c = np.zeros((feat_dim, feat_dim), dtype=np.float32)
        for i in chunk:
            feat_c = f[i][source][:]
            feat_c_ = feat_c - mean
            ATA_c += np.dot(feat_c_.T, feat_c_)
        return ATA_c

    print time.ctime(), 'start: out-of-core matrix multiplication'
    rst = Parallel(n_jobs=n_jobs)(delayed(compute_ATA)(video_names[i::n_jobs])
                                  for i in xrange(n_jobs))
    ATA = rst[0]
    for ATA_c in rst[1::]:
        ATA += ATA_c
    print time.ctime(), 'finish: out-of-core matrix multiplication'

    # SVD
    print time.ctime(), 'start: SVD in memory'
    U, S, _ = np.linalg.svd(ATA)
    print time.ctime(), 'finish: SVD in memory'

    print time.ctime(), 'serializing ...'
    hkl.dump({'x_mean': x_mean, 'U': U, 'S': S}, pcafile)


def input_parse():
    description = 'Compute PCA with A.T * A computation out of core'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('dsfile', help='HDF5-file with features')
    p.add_argument('pcafile', help='HDF5-file with PCA results')
    p.add_argument('-n', '--n_jobs', default=1, type=int,
                   help='Number of process for parallel out-of-core computations')
    return p


if __name__ == '__main__':
    p = input_parse()
    args = p.parse_args()
    main(**vars(args))
