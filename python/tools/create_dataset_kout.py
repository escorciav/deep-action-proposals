#!/usr/bin/env python
"""

Helper program to create dataset to perform model training.
It assumes that compute_priors ran succesfully and its three outputs
were dumpped on disk.

"""
import argparse
import os

import h5py
import numpy as np

from activitynet_helper import ActivityNet
from c3d_feature_helper import Feature
from data_generation import load_files
from thumos14_helper import Thumos14


def output_file_validation(dirname, prefix_list, suffix):
    ds_filename = []
    for i in prefix_list:
        ds_filename.append(os.path.join(dirname, i + suffix))
        j = ds_filename[-1]
        if os.path.exists(j):
            raise ValueError('Dataset {} already exist'.format(j))
    return ds_filename


def input_parser():
    description = ('Create hdf5-files (train/validation) with a dataset '
                   'including features of the segments and labels. It allows '
                   'to drop classes on validation set')
    p = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    h_reffile = ('CSV file dumpped by data_generation.dump_files with info '
                 'about segments')
    p.add_argument('ref_file', help=h_reffile)
    h_conffile = ('HDF5 file dumpped by data_generation.dump_files with '
                  'confidence score of every prior per segment. Required if '
                  'shuffling is passed.')
    p.add_argument('conf_file', help=h_conffile)
    p.add_argument('rootfile', help='HDF5 to retrieve features')
    p.add_argument('exp_id', help='Track dataset generated')
    p.add_argument('-sf', '--suffix_fmt', default='_fc7_{}.hkl',
                   help='Suffix format to append to dataset name')
    p.add_argument('-dset', '--dataset', default='thumos14-val',
                   help='Dataset ID: thumos14-val/test.')
    p.add_argument('-ig', '--ignore_idx', nargs='+', default=[8, 15], type=int,
                   help='idx of label to drop on validation')
    p.add_argument('output_dir', help='Fullpath of folder to place outputs')
    h_pooltype = 'Type of temporal stacking/pooling of C3D features'
    p.add_argument('-p', '--pool_type', default='mean', help=h_pooltype)
    p.add_argument('-f2', '--feat_2d', action='store_false',
                   help='Dump features as 2D matrix otherwise 3D tensor')
    p.add_argument('-s', '--shuffle', action='store_true',
                   help='Shuffle segments in order to move inertia XD')
    p.add_argument('-rng', '--rng_seed', default=None, type=int,
                   help='Integer seed for reproducibility')
    p.add_argument('-v', '--verbose', action='store_true')
    p.add_argument('-vl', '--vb_level', default=500, type=int,
                   help='Verbosity level')
    return p


def main(ref_file, conf_file, rootfile, exp_id, output_dir, suffix_fmt,
         dataset, ignore_idx, pool_type, feat_2d, shuffle, rng_seed, verbose,
         vb_level):
    rng = np.random.RandomState(rng_seed)

    dset_id, subset = dataset.lower().split('-')
    if dset_id == 'thumos14':
        th14 = Thumos14()
        df_annot = th14.segments_info(subset)
    elif dset_id == 'activitynet':
        anv12 = ActivityNet()
        df_annot = anv12.segments_info(subset)
    else:
        raise ValueError('Dataset ID not known.')

    suffix = suffix_fmt.format(pool_type) + '-' + exp_id
    dsset_name = output_file_validation(output_dir, ['train', 'val'], suffix)
    _, df, conf = load_files(ref_file=ref_file, conf_file=conf_file)

    idx_vn_rm = df_annot['label-idx'].isin(ignore_idx)
    vn_rm = df_annot.loc[idx_vn_rm, 'video-name'].unique()
    idx_bool_val = np.logical_and(np.array(df['video-name'].isin(vn_rm)),
                                  conf.sum(axis=1) > 0)
    idx_bool_train = np.logical_not(idx_bool_val)
    idx_train, idx_val = idx_bool_train.nonzero()[0], idx_bool_val.nonzero()[0]
    # Shuffling
    if shuffle:
        idx_train = rng.permutation(idx_train)
        idx_val = rng.permutation(idx_val)
    dsset = zip(dsset_name, [idx_train, idx_val])

    # Open HDF5 root dataset
    ds_root = Feature(rootfile, pool_type=pool_type)
    ds_root.open_instance()

    def hdf5_dataset_dump(filename, idx_s, df_s=df, conf_s=conf,
                          ds_feat=ds_root, feat_2d=feat_2d):
        # Create HDF5 and initialize dataset
        feat = ds_feat.read_feat(*df.loc[idx_s[0],
                                         ['video-name', 'f-init', 'duration']],
                                 return_reshaped=feat_2d)
        feat_shape = feat.shape
        if feat_2d:
            feat_shape = (feat.size,)

        f = h5py.File(filename, 'w')
        dset = f.create_dataset("data", (idx_s.size,) + feat_shape,
                                dtype=np.float32, chunks=True)
        dset.dims[0].label = 'batch'
        if feat_2d:
            dset.dims[1].label = 'feature'
        else:
            dset.dims[1].label = 't'
            dset.dims[2].label = 'feature'
        # Compatibility with previous code
        f.create_dataset('type', data=np.array(['ndarray']))

        # Loop computing features for segments and writting on disk
        colnames = ['video-name', 'f-init', 'duration']
        for i, v in enumerate(idx_s):
            feat = ds_feat.read_feat(*df.loc[v, colnames],
                                     return_reshaped=feat_2d)
            dset[i, ...] = feat

            if verbose and (i + 1) % vb_level == 0:
                print 'Processed segments {}/{}'.format(i + 1, idx_s.size)
        f.close()

        # Dump label matrix
        conffile = (filename.split('_')[0] + '_conf' +
                    os.path.splitext(filename)[1])
        with h5py.File(conffile, 'w') as f:
            dset = f.create_dataset("data", (idx_s.size, len(colnames)),
                                    dtype=np.int32, chunks=True)
            dset.dims[0].label = 'batch'
            dset.dims[1].label = 'id'
            dset[...] = np.array(conf_s[idx_s, colnames])
            # Compatibility with previous code
            f.create_dataset('type', data=np.array(['ndarray']))

    # Loop to generate train and val features
    for i in dsset:
        hdf5_dataset_dump(*i)

    # Close HDF5 root dataset
    ds_root.close_instance()


if __name__ == '__main__':
    p = input_parser()
    args = p.parse_args()
    main(**vars(args))
