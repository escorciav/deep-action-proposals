import argparse
import os

from joblib import Parallel, delayed
import pandas as pd

from utils import dump_frames


def dump_wrapper(filename, output_folder, fullpath):
    filename_noext = os.path.splitext(filename)[0]
    if fullpath:
        filename_noext = os.path.basename(filename_noext)
    output_dir = os.path.join(output_folder, filename_noext)
    return dump_frames(filename, output_dir)


def input_parse():
    description = ('Extract frames of a bucnh of videos. The file-system '
                   'organization is preserved if relative-path are used.')
    p = argparse.ArgumentParser(description=description)
    p.add_argument('input_file',
                   help='CSV file with list of videos to process')
    p.add_argument('output_folder', help='Folder to allocate frames')
    p.add_argument('-n', '--n_jobs', default=1, type=int,
                   help='Number of CPUs')
    p.add_argument('-ff', '--fullpath', action='store_true',
                   help='Input-file uses fullpath instead of rel-path')
    args = p.parse_args()
    return args


def main(input_file, output_folder, n_jobs, fullpath):
    df = pd.read_csv(input_file, sep=' ', header=None)
    Parallel(n_jobs=n_jobs)(delayed(dump_wrapper)(i, output_folder, fullpath)
                            for i in df.loc[:, 0])
    return None


if __name__ == '__main__':
    args = input_parse()
    main(**vars(args))
