#!/usr/bin/env python
"""

Helper program to compute priors/proposals

"""
import argparse

from activitynet_helper import ActivityNet
from data_generation import compute_priors, dump_files
from thumos14_helper import Thumos14


def set_dataset_helper(dataset):
    if dataset == 'activitynet':
        return ActivityNet()
    elif dataset == 'thumos14':
        return Thumos14()


def main(ds_name, outprefix, n_proposals, T, iou_thr, i_thr, rng_seed):
    ds_helper = set_dataset_helper(ds_name)
    df_seg = ds_helper.segments_info()

    # Generate segments for training and priors for regression
    priors, df = compute_priors(df_seg, T, n_proposals, iou_thr=iou_thr,
                                i_thr=i_thr, rng_seed=rng_seed)

    # Save priors, segments, confidence/matching
    dump_files(outprefix, priors=priors, df=df, conf=True)
    return None


def input_parse():
    description = ('Compute and Dump K activity priors/proposals for a length '
                   'T. It also saves sample segments of length T and match '
                   'K proposal of each segment with the respective annotation')
    p = argparse.ArgumentParser(description=description)
    p.add_argument('-k', '--n_proposals', default=16, type=int,
                   help='Number of proposals')
    p.add_argument('-t', '--T', default=256, type=int,
                   help='Segment lenght')
    p.add_argument('-iou', '--iou_thr', default=0.5, type=float,
                   help='IoU threshold')
    p.add_argument('-i', '--i_thr', default=1.0, type=float,
                   help='Intersection threshold')
    p.add_argument('-rng', '--rng_seed', default=None, type=int,
                   help='Seed for random number generator')
    p.add_argument('-d', '--ds_name', default='activitynet',
                   choices=['thumos14', 'activitynet'])
    p.add_argument('-o', '--outprefix', default='mydata_256_16',
                   help='Prefix for output files')
    return p
