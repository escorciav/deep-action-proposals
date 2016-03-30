#!/usr/bin/env python
"""

Dimensionality reduction of features

"""
import argparse
import json
import os

import h5py
import hickle as hkl
import numpy as np

PCA_SOURCE = dict(S='S', U='U', x_mean='x_mean')
DS_SOURCE = 'c3d_features'


def input_parser():
    description = 'Apply PCA over features'
    p = argparse.ArgumentParser(description=description)
    h_dsname = ('HDF5-file with features where to apply transformation')
    p.add_argument('dsfile', help=h_dsname)
    p.add_argument('pcafile', help='HDF5-file with PCA results')
    p.add_argument('-o', '--outputfile', default=None,
                   help='Fullpath name for output-file')
    g = p.add_mutually_exclusive_group()
    g.add_argument('-e', '--energy', default=0.9, type=float,
                   help='Minimium energy of eigenvalues')
    g.add_argument('-k', '--k', default=None, type=int,
                   help='Number of components to select')
    h_pcasrc = 'Dict with keys (S, U, x_mean) pointing variables of pcafile'
    p.add_argument('-ps', '--pca_src', default=PCA_SOURCE, help=h_pcasrc,
                   type=json.load)
    p.add_argument('-ds', '--ds_src', default=DS_SOURCE,
                   help='source of hdf5-file with features')
    p.add_argument('-v', '--verbose', action='store_true',
                   help='verbosity level')
    p.add_argument('-vl', '--vloop', default=100, type=int,
                   help='Control frequency of verbose level inside loops')
    return p


def main(dsfile, pcafile, outputfile=None, energy=0.9, k=None,
         pca_src=PCA_SOURCE, ds_src=DS_SOURCE, verbose=True, vloop=100):
    if outputfile is None:
        filename, ext = os.path.splitext(dsfile)
        outputfile = filename + '_pca' + ext
        if os.path.exists(outputfile):
            raise ValueError('Please provide outoutfile')

    if k is None:
        S = hkl.load(pcafile)[pca_src['S']]
        cum_energy = np.cumsum(S / S.sum())
        k = (cum_energy > energy).nonzero()[0].min() + 1
    Up = hkl.load(pcafile)[pca_src['U']][:, :k]
    x_mean = hkl.load(pcafile)[pca_src['x_mean']]

    f_raw = h5py.File(dsfile, 'r')
    f_red = h5py.File(outputfile, 'w')

    # Transform all features based on the selected components
    j, n_videos = 0, len(f_raw.keys())
    for i, v in f_raw.iteritems():
        if ds_src not in v.keys():
            j += 1
            if verbose:
                print 'Skip: video {} no dataset {}'.format(i, ds_src)
            continue

        feat = v[ds_src][:]
        feat_red = np.dot(feat - x_mean, Up)
        grp = f_red.create_group(i)
        dset = grp.create_dataset(ds_src, data=feat_red)
        j += 1
        if verbose and j % vloop == 0:
            print 'Processed videos: {}/{}'.format(j, n_videos)
    f_red.close()


if __name__ == '__main__':
    p = input_parser()
    args = p.parse_args()
    main(**vars(args))
