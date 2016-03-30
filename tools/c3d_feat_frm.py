#!/usr/bin/env python
"""

Wrapper around c3d binary tool: extract_image_features.bin

"""
import argparse
import json
import math
import os
import tempfile
from subprocess import CalledProcessError, check_output, STDOUT

import pandas as pd

import daps.utils.prototxt as prototxt

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
C3D_FEAT_EXTRACT = os.path.join(FILE_DIR, '..', '3rdparty', 'C3D', 'build',
                                'tools', 'extract_image_features.bin')


def input_parse():
    description = 'Wrapper around c3d binary tool: extract_image_features.bin'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('model', help='protofile of model')
    p.add_argument('output_csv', help='CSV output list')
    p.add_argument('layers', nargs='+', help='Layers to dump')
    p.add_argument('-p', '--proto', default='c3d_model',
                   help=('Fullpath of prototxt or str of variable in prototxt '
                         'module'))
    p.add_argument('-c', '--config', default=None, type=json.loads,
                   help=('Dict serialized as string with parameter to '
                         'configure prototxt'))
    p.add_argument('-g', '--gpu_id', default=0)
    p.add_argument('-bz', '--batch_size', default=50, type=int)
    p.add_argument('-n', '--n_batch', default=None, type=int,
                   help='Number of mini batches. Computed auto by default.')
    p.add_argument('-b', '--c3d_bin', default=C3D_FEAT_EXTRACT,
                   help='Fullpath of c3d-feature-extraction binary')
    p.add_argument('-mk', '--mkdir', action='store_true',
                   help='Make dirs for outputs')
    p.add_argument('-v', '--verbose', action='store_true',
                   help='Increase verbosity level')
    p.add_argument('-d', '--debug', action='store_true',
                   help='Not delete temporal files')
    args = p.parse_args()
    return args


def main(model, output_csv, layers, proto, config, gpu_id, batch_size,
         n_batch, c3d_bin, mkdir, verbose, debug):
    if os.path.exists(proto) and os.path.isfile(proto):
        protofile = proto
    elif proto == 'c3d_model':
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(prototxt.c3d_model.format(**config))
        protofile = f.name
    else:
        raise ValueError('Unknown file or variable {}'.format(proto))

    if not os.path.isfile(output_csv):
        raise ValueError('Unknown file {}'.format(output_csv))
    df = pd.read_csv(output_csv, header=None)
    if mkdir:
        for i in df[0]:
            dirname = os.path.dirname(i)
            if not os.path.exists(dirname):
                os.makedirs(dirname)

    if 'batch_size' in config:
        batch_size = int(config['batch_size'])
    if n_batch is None:
        n_batch = int(math.ceil(df.shape[0] * 1.0 / batch_size))

    cmd = [c3d_bin, protofile, model, str(gpu_id), str(batch_size),
           str(n_batch), output_csv] + layers
    if verbose:
        print cmd
    try:
        check_output(cmd, stderr=STDOUT)
    except CalledProcessError as e:
        print e.output

    if not debug:
        os.remove(protofile)


if __name__ == '__main__':
    args = input_parse()
    main(**vars(args))
