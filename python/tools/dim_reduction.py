import argparse
import os

import hickle as hkl
from sklearn.decomposition import RandomizedPCA as PCA


def input_parser():
    description = 'Perform Randomized PCA over features'
    p = argparse.ArgumentParser(description=description)
    h_filenames = ('HDF5-files with features where to apply transformation. '
                   'The first file is the only one use to select components.')
    p.add_argument('filenames', nargs='+', help=h_filenames)
    p.add_argument('-o', '--outputfiles', nargs='+', default=[],
                   help='Fullpath name for output-files')
    p.add_argument('-r', '--inv_ratio', default=4, dtype=int,
                   help='Inverse of ratio of selected dimensions')
    p.add_argument('-rng', '--rng_seed', default=None, dtype=int,
                   help='Seed for random number generation')
    p.add_argument('-c', '--copy', action='store_true',
                   help='Reduce memory footprint of algorithm')
    return p


def main(filenames, outputfiles=None, inv_ratio=None, copy=None,
         rng_seed=None):
    # Load train file
    X = hkl.load(filenames[0])
    n_components = X.shape[1] / inv_ratio

    prm = dict(n_components=n_components, copy=copy, random_state=rng_seed)
    pca = PCA(**prm)
    pca.fit(X)

    # Transform all features based on the selected components
    for i, v in enumerate(filenames):
        X = hkl.load(v)
        Xt = pca.transform(X)

        if len(outputfiles) == 0:
            outputfile = os.path.splitext(v)[0] + '-pca-1-{}'.format(inv_ratio)
        else:
            outputfile = outputfile[i]
        hkl.dump(Xt, outputfile)


if __name__ == '__main__':
    p = input_parser()
    args = p.parse_args()
    main(**vars(args))
