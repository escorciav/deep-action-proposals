import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from utilities import frame_rate, video_duration


def video_stats(filename):
    stats = []
    stats.append(video_duration(filename))
    stats.append(frame_rate(filename))
    return stats


def input_parse():
    description = ('Extract frames of a bucnh of videos. The file-system '
                   'organization is preserved if relative-path are used.')
    p = argparse.ArgumentParser(description=description)
    p.add_argument('input_file',
                   help='CSV file with list of videos to process')
    p.add_argument('output_file', help=('CSV-file with duration (s) and ' +
                                        'frame rate of videos'))
    p.add_argument('-n', '--n_jobs', default=1, type=int,
                   help='Number of CPUs')
    args = p.parse_args()
    return args


def main(input_file, output_file, n_jobs):
    df = pd.read_csv(input_file, sep=' ', header=None)
    stats = Parallel(n_jobs=n_jobs)(delayed(video_stats)(i)
                                    for i in df.loc[:, 0])
    df_stat = pd.DataFrame(np.array(stats))
    new_df = pd.concat((df, df_stat), axis=1)
    new_df.to_csv(output_file, sep=' ', header=None, index=False)


if __name__ == '__main__':
    args = input_parse()
    main(**vars(args))
