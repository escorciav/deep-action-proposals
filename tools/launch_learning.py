#!/usr/bin/env python
"""

Dispatch multiple learning processes

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
OPT_CHOICES = ['rmsprop', 'adam', 'adagrad', 'sgd']


def launch_jobs(pid_pool, serial_jobs, gpu, verbose, idle_time):
    # Launch jobs
    env_vars = set_device(gpu)

    for pid_name, pid_attb in pid_pool.iteritems():
        if verbose:
            print pid_attb[0]

        pid_attb[1] = Popen(pid_attb[0], env=env_vars)

        # Polling scheme serial jobs
        if serial_jobs:
            while pid_attb[1].poll() is None:
                time.sleep(idle_time)

            print 'ID {} finished'.format(pid_name)

    # Polling scheme parallel jobs
    while not serial_jobs:
        pid_names = pid_pool.keys()
        for pid in pid_names:
            if pid_pool[pid][1].poll() is not None:
                print 'ID {} finished'.format(pid)
                pid_pool.pop(pid)

        if len(pid_pool) == 0:
            print 'All process finished. Bye!'
            break
        else:
            time.sleep(idle_time)


def set_device(gpu):
    # Set device ID
    if gpu >= 0:
        dev_flags = 'device=gpu' + str(gpu)
    else:
        dev_flags = 'device=cpu'
    env_vars = os.environ
    env_vars['THEANO_FLAGS'] = dev_flags
    return env_vars


def set_model(model_type, num_proposal=None, depth=None, width=None,
              seq_length=None, drop_in=None, drop_out=None):
    # Set model string
    if model_type.lower() == 'mlp':
        model_fmt = 'mlp:{},{},{},{},{}'
        model = model_fmt.format(int(num_proposal), int(depth), int(width),
                                 drop_in, drop_out)
    elif model_type.lower() == 'lstm':
        model_fmt = 'lstm:{},{},{},{}'
        model = model_fmt.format(int(num_proposal), int(seq_length),
                                 int(width), int(depth))
    return model


def main(id_fmt, id_offset, model_type, num_proposal, depth, width,
         seq_length, drop_in, drop_out, grad_clip, forget_bias, batch_size,
         n_epoch, l_rate, w_pos, alpha, beta, opt_rule, opt_prm, reg, rng_seed,
         init_model, shuffle, snapshot_freq, output_dir, ds_prefix, ds_suffix,
         debug, gpu, serial_jobs, idle_time, verbose):
    # Set dir for logs, snapshots, etc.
    if output_dir is None:
        output_dir = ds_prefix

    # Opt parameter
    opt_prm = []
    if opt_prm:
        opt_prm = ['-op', opt_prm]

    # Seed for pseudo-random number generator
    rng_prm = []
    if rng_seed:
        rng_prm = ['-rng', str(rng_seed)]

    # Shuffling
    shuffle_prm = []
    if shuffle:
        shuffle_prm = ['-sh']

    # Include init_model to reinitialize
    include_init_model = []
    if init_model:
        include_init_model = ['-i', str(init_model[0]), str(init_model[1])]

    # Debug mode
    debug_mode = []
    if debug:
        debug_mode = ['-dg']

    opt_id = [i for i, v in enumerate(OPT_CHOICES) if v in opt_rule]
    # Cartesian product
    prm = np.vstack(map(lambda x: x.flatten(),
                        np.meshgrid(l_rate, alpha, depth, width, opt_id, w_pos,
                                    beta, indexing='ij')))

    # Make cmd
    pid_pool = {}
    for i in range(prm.shape[1]):
        exp_id = id_fmt.format(i + id_offset)
        model = set_model(model_type, num_proposal, prm[2, i], prm[3, i],
                          seq_length, drop_in, drop_out)
        cmd = (['python', 'daps/learning.py', '-id', exp_id, '-m', model,
                '-a', str(prm[1, i]), '-ne', str(n_epoch), '-od', output_dir,
                '-lr', str(prm[0, i]), '-dp', ds_prefix, '-ds', ds_suffix,
                '-sf', str(snapshot_freq), '-bz', str(batch_size), '-w+',
                str(prm[5, i]), '-om', OPT_CHOICES[prm[4, i].astype(int)],
                '-gc', str(grad_clip), '-r', reg, '-b', str(prm[6, i]),
                '-fb', str(forget_bias)] + include_init_model + opt_prm +
               rng_prm + debug_mode + shuffle_prm)
        pid_pool[exp_id] = [cmd, None]

    launch_jobs(pid_pool, serial_jobs, gpu, verbose, idle_time)
    print "My job is done. Good luck with your results!"


if __name__ == '__main__':
    h_idoffset = 'Offset for exp-id. Used it output_dir is the same'
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-np', '--num_proposal', default=16,
                   help='Number of proposals/priors')
    p.add_argument('-d', '--depth', default=2, nargs='+', type=int,
                   help='Num FC layers')
    p.add_argument('-w', '--width', default=5000, nargs='+', type=int,
                   help='No hidden units')
    p.add_argument('-l', '--seq_length', default=16, help='LSTM length')
    p.add_argument('-din', '--drop_in', default=0, help='Dropout inputs')
    p.add_argument('-dout', '--drop_out', default=0.5, help='Dropout hidden')
    p.add_argument('-ne', '--n_epoch', default=200, type=int,
                   help='Num epochs')
    p.add_argument('-bz', '--batch_size', default=500, type=int,
                   help='Mini batch size')
    p.add_argument('-gc', '--grad_clip', default=100, type=float,
                   help='Gradient clipping')
    p.add_argument('-fb', '--forget_bias', default=0, type=float,
                   help='Set bias of forget gate on LSTM')
    p.add_argument('-lr', '--l_rate', default=L_RATE, nargs='+', type=float,
                   help='List of learning rate values')
    p.add_argument('-a', '--alpha', default=ALPHA, nargs='+', type=float,
                   help='List of alpha values')
    p.add_argument('-b', '--beta', default=0.0, nargs='+', type=float,
                   help='Regularizer contribution')
    p.add_argument('-w+', '--w_pos', default=1.0, nargs='+', type=float,
                   help='Weigth for positive samples on loss function')
    p.add_argument('-or', '--opt_rule', nargs='+', default='sgd',
                   choices=OPT_CHOICES, help='Optimization method')
    p.add_argument('-op', '--opt_prm', default=None,
                   help='Parameters of optimization method')
    p.add_argument('-r', '--reg', default='l2', choices=['l1', 'l2'],
                   help='Type of regularizer penalty')
    h_initmodel = ('Pair of model-path, epoch to restart learning from this '
                   'point')
    p.add_argument('-rng', '--rng_seed', default=None, type=int,
                   help='Seed random number generator')
    p.add_argument('-i', '--init_model', nargs=2, default=None,
                   help=h_initmodel)
    p.add_argument('-sh', '--shuffle', action='store_true',
                   help='Shuffle data samples at every iteration')
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
    p.add_argument('-dg', '--debug', action='store_true',
                   help='Run learning.py on debug mode')
    p.add_argument('-g', '--gpu', default=0, type=int, help='Device ID')
    p.add_argument('-sj', '--serial_jobs', action='store_true',
                   help='Launch jobs serially')
    p.add_argument('-s', '--idle_time', default=60*5, type=int,
                   help='Idle time between polling stages')
    p.add_argument('-v', '--verbose', action='store_true')
    args = p.parse_args()
    main(**vars(args))
