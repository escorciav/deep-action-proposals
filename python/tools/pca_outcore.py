#!/usr/bin/env python
"""

PCA done via matrix multiplication out-of-core.

"""
import argparse
import time

import h5py
import hickle as hkl
import numpy as np


def main(dsfile, pcafile, t_size=16, t_stride=8, source='c3d_features',
         log_loop=100):
    print time.ctime(), 'start: loading hdf5'
    fid = h5py.File(dsfile, 'r')
    video_names = fid.keys()
    feat_dim = fid[video_names[0]][source].shape[1]
    print time.ctime(), 'finish: loading hdf5'

    print time.ctime(), 'start: compute mean'
    x_mean, n = np.zeros((1, feat_dim), dtype=np.float32), 0
    for i, v in fid.iteritems():
        feat = v[source][:]
        n += feat.shape[0]
        x_mean += feat.sum(axis=0)
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
    j, n_videos = 0, len(video_names)
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
    hkl.dump({'x_mean': x_mean, 'U': U, 'S': S, 'n_samples': n}, pcafile)


def input_parse():
    description = 'Compute PCA with A.T * A computation out of core'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('dsfile', help='HDF5-file with features')
    p.add_argument('pcafile', help='HDF5-file with PCA results')
    p.add_argument('-ll', '--log_loop', default=500, type=int,
                   help='Verbose in terms of number of videos')
    return p


if __name__ == '__main__':
    p = input_parse()
    args = p.parse_args()
    main(**vars(args))
