#!/usr/bin/env python
"""

Wrapper to dispatch several learners

"""
import argparse
import os
import time
from subprocess import Popen

import numpy as np

OUTPUT_DIR = 'data/experiments/thumos14/a01'
L_RATE = [1e-2, 1e-3, 1e-4, 1e-5]
ALPHA = [1e-6, 5e-2, 1e-1, 0.3]
MODEL_CHOICES = ['lstm', 'mlp']


def main(num_proposal, depth, width, seq_length, drop_in, drop_out,
         n_epoch, l_rate, alpha, init_model,
         id_fmt, id_offset, model_type, gpu, snapshot_freq,
         output_dir, ds_prefix, ds_suffix, idle_time):
    # Set dir for logs, snapshots, etc.
    if output_dir is None:
        output_dir = ds_prefix

    # Set device ID
    if gpu >= 0:
        dev_flags = 'device=gpu' + str(gpu)
    else:
        dev_flags = 'device=cpu'
    env_vars = os.environ
    env_vars['THEANO_FLAGS'] = dev_flags

    # Set model string
    if model_type.lower() == 'mlp':
        model_fmt = 'mlp:{},{},{},{},{}'
        model = model_fmt.format(num_proposal, depth, width, drop_in, drop_out)
    elif model_type.lower() == 'lstm:':
        model_fmt = 'lstm:{},{},{},{}'
        model = model_fmt.format(num_proposal, seq_length, width, depth)

    # Include init_model to reinitialize
    include_init_model = []
    if init_model:
        include_init_model = ['-i', str(init_model[0]), str(init_model[1])]

    # Cartesian product
    prm = np.dstack(np.meshgrid(l_rate, alpha)).reshape(-1, 2)

    # Launch process
    pid_pool = {}
    for i in range(prm.shape[0]):
        exp_id = id_fmt.format(i + id_offset)
        cmd = ['python', 'python/learning.py', '-id', exp_id, '-m', model,
               '-a', str(prm[i, 1]), '-ne', str(n_epoch), '-od', output_dir,
               '-lr', str(prm[i, 0]), '-dp', ds_prefix, '-ds', ds_suffix,
               '-sf', str(snapshot_freq)] + include_init_model
        print cmd
        pid_pool[exp_id] = Popen(cmd, env=env_vars)

    # Polling
    while True:
        pid_names = pid_pool.keys()
        for pid in pid_names:
            if pid_pool[pid].poll() is not None:
                print 'ID {} finished'.format(pid)
                pid_pool.pop(pid)

        if len(pid_pool) == 0:
            print 'All process finished. Bye!'
            break
        else:
            time.sleep(idle_time)


if __name__ == '__main__':
    h_idoffset = 'Offset for exp-id. Used it output_dir is the same'
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-np', '--num_proposal', default=64,
                   help='Number of proposals/priors')
    p.add_argument('-d', '--depth', default=2, help='Num FC layers')
    p.add_argument('-w', '--width', default=5000, help='No hidden units')
    p.add_argument('-l', '--seq_length', default=16, help='LSTM length')
    p.add_argument('-din', '--drop_in', default=0, help='Dropout inputs')
    p.add_argument('-dout', '--drop_out', default=0.5, help='Dropout hidden')
    p.add_argument('-ne', '--n_epoch', default=200, type=int,
                   help='Num epochs')
    p.add_argument('-lr', '--l_rate', default=L_RATE, nargs='+', type=float,
                   help='List of learning rate values')
    p.add_argument('-a', '--alpha', default=ALPHA, nargs='+', type=float,
                   help='List of alpha values')
    h_initmodel = ('Pair of model-path, epoch to restart learning from this '
                   'point')
    p.add_argument('-i', '--init_model', nargs=2, default=None,
                   help=h_initmodel)
    p.add_argument('-sf', '--snapshot_freq', default=60, type=int,
                   help='Frequency of snapshots')
    p.add_argument('-if', '--id_fmt', default='{:03d}', help='Exp ID format')
    p.add_argument('-io', '--id_offset', default=0, type=int, help=h_idoffset)
    p.add_argument('-mt', '--model_type', default='mlp', choices=MODEL_CHOICES,
                   help='Type of architecture')
    p.add_argument('-dp', '--ds_prefix', type=str, default=OUTPUT_DIR)
    p.add_argument('-ds', '--ds_suffix', type=str, default='mean')
    p.add_argument('-od', '--output_dir', default=None,
                   help='Folder to allocate experiment results')
    p.add_argument('-g', '--gpu', default=0, type=int, help='Device ID')
    p.add_argument('-s', '--idle_time', default=60*5, type=int,
                   help='Idle time between polling stages')
    args = p.parse_args()
    main(**vars(args))
