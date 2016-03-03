import argparse
import glob
import hickle as hkl
import json
import numpy as np
import os
import pandas as pd

from model import build_model
from model import read_model
from eval_model import nms_detections
from eval_model import retrieve_proposals
from utils import segment_format
from activitynet_helper import ActivityNet
from thumos14_helper import Thumos14


def filter_proposals(proposal_df):
    """Remove non-coherent proposals from DataFrame.
    """
    neg_idx = proposal_df['f-init'] < 0
    non_causal_idx = proposal_df['f-end'] <= proposal_df['f-init']
    idx = ~(neg_idx | non_causal_idx)
    proposal_df = proposal_df.loc[idx].reset_index(drop=True)
    return proposal_df


def load_proposals(proposal_dir, stride=128, T=256,
                   file_filter=None, priors_filename=None):
    """Load proposal DataFrames from files.
    """
    proposal_df = []
    vds_true = None
    if file_filter:
        vds_true = pd.read_csv(file_filter)['video-name'].tolist()
    filenames = glob.glob(os.path.join(proposal_dir, '*.proposals'))
    priors = None
    if priors_filename:
        priors = hkl.load(priors_filename)
    for f in filenames:
        vid = os.path.basename(f).split('.')[0]
        if file_filter and vid not in vds_true:
            continue
        this_df = pd.read_csv(f, sep=' ', index_col=False)
        if priors_filename:
            n_proposals = priors.shape[0]
            n_segments = this_df.shape[0] / n_proposals
            this_priors = np.tile(priors, (n_segments, 1))
            l_size = this_df['video-frames'].mean()
            f_init_array = np.arange(0, l_size - T, stride)
            map_array = np.stack((f_init_array, np.zeros(n_segments)))
            map_array = map_array.repeat(n_proposals, axis=-1).T
            proposals = segment_format(
                map_array + (this_priors.clip(0, 1) * T), 'c2b').astype(int)
            this_df['f-init'] = proposals[:, 0]
            this_df['f-end'] = proposals[:, 1]
        proposal_df.append(this_df)
    return pd.concat(proposal_df, axis=0)


def wrapper_nms(proposal_df, overlap=0.65):
    """Apply non-max-suppresion to a video batch.
    """
    vds_unique = pd.unique(proposal_df['video-name'])
    new_proposal_df = []
    for i, v in enumerate(vds_unique):
        idx = proposal_df['video-name'] == v
        p = proposal_df.loc[idx, ['video-name', 'f-init', 'f-end',
                                  'score', 'video-frames']]
        n_frames = np.int(p['video-frames'].mean())
        loc = np.stack((p['f-init'], p['f-end']), axis=-1)
        loc, score = nms_detections(loc, np.array(p['score']), overlap)
        n_proposals = score.shape[0]
        n_frames = np.repeat(p['video-frames'].mean(), n_proposals).astype(int)
        this_df = pd.DataFrame({'video-name': np.repeat(v, n_proposals),
                                'f-init': loc[:, 0], 'f-end': loc[:, 1],
                                'score': score,
                                'video-frames': n_frames})
        new_proposal_df.append(this_df)
    return pd.concat(new_proposal_df, axis=0)


def wrapper_retrieve_proposals(video_df, network, proposal_dir, T=256,
                               stride=128, c3d_size=16, c3d_stride=8,
                               pool_type='mean', hdf5_dataset=None,
                               model_prm=None, verbose=True):
    """Retrieve proposals for a video batch and save them.
    """
    cnt = 1
    for idx, video in video_df.iterrows():
        if video['video-frames'] < T:
            cnt += 1
            continue
        proposals, score = retrieve_proposals(
            video['video-name'], video['video-frames'], network, T, stride,
            c3d_size, c3d_stride, pool_type, hdf5_dataset, model_prm)

        # Build proposal DataFrame.
        n_proposals = proposals.shape[0]
        this_proposal_df = pd.DataFrame(
            {'video-name': np.repeat(video['video-name'], n_proposals),
             'video-frames': np.repeat(video['video-frames'], n_proposals),
             'f-init': proposals[:, 0], 'f-end': proposals[:, 1],
             'score': score})
        out = os.path.join(proposal_dir,
                           '{}.proposals'.format(video['video-name']))
        this_proposal_df.to_csv(out, sep=' ', index=False,
                                columns=['video-name', 'video-frames',
                                         'f-init', 'f-end', 'score'])
        if verbose:
            print 'Processed video: {} - {}/{}'.format(video['video-name'],
                                                       cnt, video.shape[0])
            cnt += 1


def input_parser():
    description = 'Evaluates a deep action proposal model.'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('model',
                   help='npz file with trained model.')
    p.add_argument('network_params',
                   help='json file with params used for the input model')
    p.add_argument('eval_id', help='Evaluation identifier')
    p.add_argument('exp_id',
                   help='Experiment identifier to correlate with train exps.')
    p.add_argument('output_dir', help='Root output directory.')
    p.add_argument('-is', '--input_size', type=int, default=4096,
                   help='Size of the input to the network.')
    p.add_argument('-dset', '--dataset', default='thumos14-val',
                   help='Dataset ID: thumos14-val/test.')
    p.add_argument('-feat', '--feat_file',
                   help='hdf5 file containing the raw C3D features.')
    p.add_argument('-ff', '--file_filter',
                   help='File containing a list of videos to be evaluated')
    p.add_argument('-ow', '--overwrite', type=bool, default=False,
                   help='Overwrite results.')
    p.add_argument('-c3d-size', '--c3d_size', type=int, default=16,
                   help='Size of C3D receptive field.')
    p.add_argument('-c3d-stride', '--c3d_stride', type=int, default=8,
                   help='Size of sliding step for C3D.')
    p.add_argument('-pt', '--pool_type', default='mean',
                   help='Type of pooling: mean, max, pyr-2-mean ...')
    p.add_argument('-s', '--stride', type=int, default=128,
                   help='Video sliding step size.')
    p.add_argument('-t', '--T', type=int, default=256,
                   help='Segment canonical size.')
    p.add_argument('-nms', '--nms', type=bool, default=True,
                   help='Apply nms to retrieved proposals')
    p.add_argument('-pr', '--priors_filename',
                   help='File with priors used in training.')
    return p


def main(model, network_params, eval_id, exp_id, output_dir,
         input_size=4096, dataset='thumos14-val', feat_file=None,
         file_filter=None, overwrite=False, c3d_size=16, c3d_stride=8,
         pool_type='mean', stride=128, T=256, nms=True, priors_filename=None):

    ###########################################################################
    # Loading dataset info.
    ###########################################################################
    # Defining dataset.
    dset_id, subset = dataset.lower().split('-')
    if dset_id == 'thumos14':
        th14 = Thumos14()
        df = th14.segments_info(subset)
    elif dset_id == 'activitynet':
        anv12 = ActivityNet()
        df = anv12.segments_info(subset)
    else:
        raise ValueError('Dataset ID not known.')
    # Defining feature path.
    if not feat_file:
        feat_file = 'data/{}/c3d/{}_c3d_temporal.hdf5'.format(dset_id, subset)
        if not os.path.exists(feat_file):
            raise ValueError('Feature file does not exists.')
    # Annotations from specific videos.
    if file_filter:
        video_names = pd.read_csv(file_filter)['video-name']
        df = df[df['video-name'].isin(video_names)]

    ###########################################################################
    # Set output file paths.
    ###########################################################################
    output_dir = os.path.join(output_dir, exp_id, 'eval', eval_id)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    # Config file.
    conf = {'stride': stride, 'nms': nms, 'dataset': dataset, 'T': T,
            'filter': file_filter, 'priors_filename': priors_filename}
    conf_filename = os.path.join(output_dir, 'config.json')
    with open(conf_filename, 'w') as fobj:
        fobj.write(json.dumps(conf, sort_keys=True, indent=4))
    # Result filename.
    result_filename = os.path.join(output_dir, 'result.proposals')
    # Proposal result path.
    proposal_dir = os.path.join(output_dir, 'proposals')
    if not os.path.isdir(proposal_dir):
        os.makedirs(proposal_dir)
    # Results path.
    results_dir = os.path.join(output_dir, 'results')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    ###########################################################################
    # Loading network.
    ###########################################################################
    with open(network_params, 'r') as fobj:
        network_params = json.load(fobj)
    network = build_model(model_prm=network_params['model'],
                          input_size=input_size)
    read_model(model, network)

    ###########################################################################
    # Proposal extraction for batch of videos.
    ###########################################################################
    vds_unique = pd.unique(df['video-name'])
    video_frames = []
    video_names = []
    # Look for existing files.
    files = glob.glob(os.path.join(proposal_dir, '*.proposals'))
    vds_true = [os.path.basename(f).split('.proposals')[0] for f in files]
    for i, vds in enumerate(vds_unique):
        idx = df['video-name'] == vds
        # Avoid recomputing if desired.
        if not overwrite and vds in vds_true:
            continue
        else:
            video_frames.append(int(df.loc[idx, 'video-frames'].mean()))
            video_names.append(vds)
    video_df = pd.DataFrame({'video-name': np.array(video_names),
                             'video-frames': np.array(video_frames)})

    if video_df.shape[0]:
        wrapper_retrieve_proposals(video_df, network, proposal_dir, T=T,
                                   stride=stride,
                                   c3d_size=c3d_size, c3d_stride=c3d_stride,
                                   pool_type=pool_type, hdf5_dataset=feat_file,
                                   model_prm=network_params['model'])

    ###########################################################################
    # Evaluate proposals
    ###########################################################################
    proposal_df = load_proposals(proposal_dir, stride=stride, T=T,
                                 file_filter=file_filter,
                                 priors_filename=priors_filename)
    if nms:
        proposal_df = wrapper_nms(proposal_df)
    # Store results
    proposal_df.to_csv(result_filename, sep=' ', index=False)
    print 'Have a good day!'


if __name__ == '__main__':
    p = input_parser()
    args = vars(p.parse_args())
    main(**args)
